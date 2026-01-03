"""
generate_data.py - Synthetic Data Generation for OpenTab Training

This module implements the SCM-based synthetic data generation approach from the TabPFN paper.
The approach uses Structural Causal Models (SCMs) to generate diverse synthetic tabular datasets
that capture characteristics of real-world data.

Key Components:
1. Graph Structure Sampling: Growing network with redirection (preferential attachment)
2. Computational Edge Mappings: Neural networks, decision trees, categorical discretization
3. Initialization Data Sampling: Normal, uniform, mixed with prototype-based non-independence
4. Post-processing: Kumaraswamy warping, quantization, missing values

Reference:
- "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"
"""

import argparse
import random
import os
from typing import Tuple, Optional, Dict, Callable, List
from dataclasses import dataclass, field
from functools import partial
import multiprocessing as mp

import numpy as np
import h5py
from tqdm import tqdm


@dataclass
class SyntheticDataset:
    """A single synthetic dataset."""
    X: np.ndarray  # (n_samples, n_features)
    y: np.ndarray  # (n_samples,)
    train_size: int  # Number of training samples
    n_classes: int  # Number of classes (for classification), 0 for regression
    is_regression: bool = False
    categorical_mask: Optional[np.ndarray] = None  # (n_features,) bool
    missing_mask: Optional[np.ndarray] = None  # (n_samples, n_features) bool
    n_categories: Optional[np.ndarray] = None  # (n_features,) int


@dataclass
class SCMHyperparameters:
    """
    High-level hyperparameters governing the synthetic dataset properties.
    Sampled at the start of each dataset generation.
    """
    # Graph structure
    n_nodes: int = 20  # Number of nodes in the DAG
    redirection_prob: float = 0.3  # Probability of edge redirection in preferential attachment
    n_subgraphs: int = 1  # Number of disjoint subgraphs to merge
    
    # Node dimensions
    node_dim: int = 8  # Dimension of vectors at each node
    
    # Dataset size
    n_samples: int = 100
    n_features: int = 10
    n_classes: int = 2  # For classification; 0 for regression
    
    # Initialization
    init_type: str = 'normal'  # 'normal', 'uniform', or 'mixed'
    init_scale: float = 1.0
    prototype_fraction: float = 0.0  # Fraction of samples as prototypes (0 = independent)
    prototype_temperature: float = 1.0  # Temperature for prototype mixing
    
    # Edge mappings
    edge_noise_std: float = 0.1
    
    # Post-processing
    apply_kumaraswamy: bool = False
    kumaraswamy_a: float = 1.0
    kumaraswamy_b: float = 1.0
    quantization_prob: float = 0.0
    missing_prob: float = 0.0


def sample_hyperparameters(
    n_samples_range: Tuple[int, int] = (10, 512),
    n_features_range: Tuple[int, int] = (1, 160),
    n_classes_range: Tuple[int, int] = (2, 10),
    node_dim_range: Tuple[int, int] = (4, 16),
    is_regression: bool = False,
    max_cells: int = 75000,
) -> SCMHyperparameters:
    """
    Sample high-level hyperparameters for dataset generation.
    Following the paper, hyperparameters are sampled from specific distributions.
    
    Paper specifications:
    - n_samples: uniform up to 2048 (we use configurable max, default 512)
    - n_features: Beta(0.95, 8.0) scaled to [1, 160]
    - max_cells: 75,000 (reduce samples if n_samples * n_features exceeds this)
    """
    # Graph size: log-uniform distribution
    n_nodes_min, n_nodes_max = 10, 50
    n_nodes = int(np.exp(random.uniform(np.log(n_nodes_min), np.log(n_nodes_max))))
    
    # Redirection probability: Gamma distribution
    # Smaller values lead to denser graphs
    alpha, beta = 2.0, 5.0
    redirection_prob = min(0.9, np.random.gamma(alpha, 1/beta))
    
    # Number of subgraphs (sometimes features are independent of target)
    n_subgraphs = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
    
    # Node dimension
    node_dim = random.randint(*node_dim_range)
    
    # Dataset properties
    # Paper: n_samples uniform up to max
    n_samples = random.randint(*n_samples_range)
    
    # Paper: n_features from Beta(k=0.95, b=8.0) scaled to [1, 160]
    beta_sample = np.random.beta(0.95, 8.0)  # Sample in [0, 1]
    n_features_min, n_features_max = n_features_range
    n_features = int(beta_sample * (n_features_max - n_features_min) + n_features_min)
    n_features = max(n_features_min, min(n_features_max, n_features))
    
    # Paper: Cap table size at 75,000 cells by reducing samples for large feature counts
    if n_samples * n_features > max_cells:
        n_samples = max(1, max_cells // n_features)
    n_classes = 0 if is_regression else random.randint(*n_classes_range)
    
    # Initialization type
    init_type = random.choice(['normal', 'uniform', 'mixed'])
    init_scale = random.uniform(0.5, 2.0)
    
    # Prototype-based non-independence (occasionally)
    if random.random() < 0.3:
        prototype_fraction = random.uniform(0.1, 0.5)
        prototype_temperature = random.uniform(0.1, 2.0)
    else:
        prototype_fraction = 0.0
        prototype_temperature = 1.0
    
    # Edge noise
    edge_noise_std = random.uniform(0.01, 0.3)
    
    # Post-processing probabilities
    # Paper: Kumaraswamy applied to "some datasets" - we use 20% of datasets, 50% of features within
    apply_kumaraswamy = random.random() < 0.2
    kumaraswamy_a = random.uniform(0.5, 2.0) if apply_kumaraswamy else 1.0
    kumaraswamy_b = random.uniform(0.5, 2.0) if apply_kumaraswamy else 1.0
    quantization_prob = random.uniform(0.0, 0.5) if random.random() < 0.4 else 0.0
    missing_prob = random.uniform(0.0, 0.3) if random.random() < 0.3 else 0.0
    
    return SCMHyperparameters(
        n_nodes=n_nodes,
        redirection_prob=redirection_prob,
        n_subgraphs=n_subgraphs,
        node_dim=node_dim,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        init_type=init_type,
        init_scale=init_scale,
        prototype_fraction=prototype_fraction,
        prototype_temperature=prototype_temperature,
        edge_noise_std=edge_noise_std,
        apply_kumaraswamy=apply_kumaraswamy,
        kumaraswamy_a=kumaraswamy_a,
        kumaraswamy_b=kumaraswamy_b,
        quantization_prob=quantization_prob,
        missing_prob=missing_prob,
    )


# ============================================================================
# Graph Structure Sampling
# ============================================================================

def sample_dag_growing_network(n_nodes: int, redirection_prob: float) -> np.ndarray:
    """
    Sample a DAG using the growing network with redirection method.
    
    This is a preferential attachment process that generates scale-free networks.
    Reference: Krapivsky & Redner (2001)
    
    Args:
        n_nodes: Number of nodes in the graph
        redirection_prob: Probability of redirecting an edge to the target's parent
        
    Returns:
        Adjacency matrix (n_nodes, n_nodes) where adj[i,j]=1 means edge from j to i
    """
    adj = np.zeros((n_nodes, n_nodes))
    
    if n_nodes < 2:
        return adj
    
    # Track parents for each node (for redirection)
    parents = [[] for _ in range(n_nodes)]
    
    for i in range(1, n_nodes):
        # Pick a random existing node to connect to
        target = random.randint(0, i - 1)
        
        # With probability redirection_prob, redirect to one of target's parents
        if parents[target] and random.random() < redirection_prob:
            target = random.choice(parents[target])
        
        # Add edge from target to new node i
        adj[i, target] = 1
        parents[i].append(target)
        
        # Occasionally add more edges (to make graph denser)
        n_extra_edges = np.random.poisson(0.5)
        for _ in range(n_extra_edges):
            potential_target = random.randint(0, i - 1)
            if adj[i, potential_target] == 0:
                adj[i, potential_target] = 1
                parents[i].append(potential_target)
    
    return adj


def sample_dag_with_subgraphs(
    n_nodes: int,
    redirection_prob: float,
    n_subgraphs: int,
) -> np.ndarray:
    """
    Sample a DAG that may consist of multiple disjoint subgraphs.
    
    Disjoint subgraphs lead to features that are marginally independent
    of the target if not connected to the target node.
    """
    if n_subgraphs <= 1:
        return sample_dag_growing_network(n_nodes, redirection_prob)
    
    # Divide nodes among subgraphs
    nodes_per_subgraph = n_nodes // n_subgraphs
    adj = np.zeros((n_nodes, n_nodes))
    
    start = 0
    for s in range(n_subgraphs):
        end = start + nodes_per_subgraph if s < n_subgraphs - 1 else n_nodes
        subgraph_size = end - start
        
        if subgraph_size > 1:
            sub_adj = sample_dag_growing_network(subgraph_size, redirection_prob)
            adj[start:end, start:end] = sub_adj
        
        start = end
    
    return adj


# ============================================================================
# Activation Functions for Edge Mappings
# ============================================================================

ACTIVATION_FUNCTIONS = {
    # Paper: identity, logarithm, sigmoid, absolute, sine, tanh, rank, squaring, power, smooth ReLU, step, modulo
    'identity': lambda x: x,
    'relu': lambda x: np.maximum(0, x),
    'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
    'tanh': np.tanh,
    'sin': np.sin,
    'abs': np.abs,
    'square': lambda x: x ** 2,
    'sqrt_abs': lambda x: np.sqrt(np.abs(x) + 1e-8),
    'log': lambda x: np.log(np.abs(x) + 1e-8),  # logarithm from paper
    'step': lambda x: (x > 0).astype(float),
    'softplus': lambda x: np.log1p(np.exp(np.clip(x, -20, 20))),  # smooth ReLU from paper
    'modulo': lambda x: np.mod(x, 1.0),
    'power_2': lambda x: np.clip(x, -10, 10) ** 2,
    'power_3': lambda x: np.clip(x, -10, 10) ** 3,
    'power_4': lambda x: np.clip(x, -10, 10) ** 4,
    'power_5': lambda x: np.clip(x, -10, 10) ** 5,
    'rank': lambda x: np.argsort(np.argsort(x, axis=0), axis=0).astype(float) / (x.shape[0] - 1 + 1e-8),
}


def get_random_activation() -> Callable:
    """Sample a random activation function."""
    name = random.choice(list(ACTIVATION_FUNCTIONS.keys()))
    return ACTIVATION_FUNCTIONS[name]


# ============================================================================
# Computational Edge Mappings
# ============================================================================

class NeuralNetworkMapping:
    """
    Small neural network as edge mapping.
    
    Applies a linear transformation followed by an element-wise nonlinearity.
    Uses Xavier initialization for weights.
    """
    
    def __init__(self, input_dim: int, output_dim: int, n_hidden_layers: int = 1):
        self.layers = []
        
        dims = [input_dim]
        for _ in range(n_hidden_layers):
            dims.append(random.randint(input_dim, max(input_dim, output_dim)))
        dims.append(output_dim)
        
        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            W = np.random.randn(dims[i], dims[i+1]) * scale
            b = np.random.randn(dims[i+1]) * scale * 0.1
            activation = get_random_activation() if i < len(dims) - 2 else ACTIVATION_FUNCTIONS['identity']
            self.layers.append((W, b, activation))
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the neural network mapping. x: (n_samples, input_dim)"""
        h = x
        for W, b, activation in self.layers:
            h = h @ W + b
            h = activation(h)
        return h


class DecisionTreeMapping:
    """
    Decision tree as edge mapping.
    
    Implements rule-based dependencies by selecting features and
    applying thresholds to determine the output.
    """
    
    def __init__(self, input_dim: int, output_dim: int, max_depth: int = 4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tree = self._build_tree(max_depth, depth=0)
    
    def _build_tree(self, max_depth: int, depth: int) -> Dict:
        """Recursively build a random decision tree."""
        if depth >= max_depth or random.random() < 0.3:
            # Leaf: random output vector
            return {'type': 'leaf', 'value': np.random.randn(self.output_dim)}
        
        # Internal node: split on a random feature dimension
        feature_idx = random.randint(0, self.input_dim - 1)
        threshold = random.uniform(-2, 2)
        
        return {
            'type': 'split',
            'feature': feature_idx,
            'threshold': threshold,
            'left': self._build_tree(max_depth, depth + 1),
            'right': self._build_tree(max_depth, depth + 1),
        }
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the decision tree. x: (n_samples, input_dim)"""
        n_samples = x.shape[0]
        output = np.zeros((n_samples, self.output_dim))
        
        for i in range(n_samples):
            output[i] = self._evaluate(self.tree, x[i])
        
        return output
    
    def _evaluate(self, node: Dict, x: np.ndarray) -> np.ndarray:
        """Evaluate tree on a single sample."""
        if node['type'] == 'leaf':
            return node['value']
        
        if x[node['feature']] < node['threshold']:
            return self._evaluate(node['left'], x)
        else:
            return self._evaluate(node['right'], x)


class CategoricalDiscretization:
    """
    Categorical feature discretization via nearest neighbor.
    
    Maps continuous vectors to the index of the nearest prototype,
    then embeds that index back to a continuous vector.
    """
    
    def __init__(self, input_dim: int, output_dim: int, n_categories: int = None):
        if n_categories is None:
            # Sample number of categories from gamma distribution with offset
            n_categories = int(np.random.gamma(2, 2)) + 2
        self.n_categories = min(n_categories, 10)  # Max 10 classes per paper
        
        # Random prototype vectors for each category
        self.prototypes = np.random.randn(self.n_categories, input_dim)
        
        # Embedding vectors to continue propagation
        self.embeddings = np.random.randn(self.n_categories, output_dim)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply discretization. x: (n_samples, input_dim)"""
        # Find nearest prototype for each sample
        # distances: (n_samples, n_categories)
        distances = np.sum((x[:, np.newaxis, :] - self.prototypes[np.newaxis, :, :]) ** 2, axis=2)
        category_idx = np.argmin(distances, axis=1)
        
        # Return embeddings for the assigned categories
        return self.embeddings[category_idx]
    
    def get_categories(self, x: np.ndarray) -> np.ndarray:
        """Return category indices for each sample."""
        distances = np.sum((x[:, np.newaxis, :] - self.prototypes[np.newaxis, :, :]) ** 2, axis=2)
        return np.argmin(distances, axis=1)


def sample_edge_mapping(input_dim: int, output_dim: int) -> Callable:
    """
    Sample a random computational edge mapping.
    
    Types (from paper):
    1. Small neural networks with various activations
    2. Decision trees
    3. Categorical discretization
    """
    mapping_type = random.choices(
        ['neural_network', 'decision_tree', 'categorical'],
        weights=[0.6, 0.25, 0.15]
    )[0]
    
    if mapping_type == 'neural_network':
        n_layers = random.randint(1, 3)
        return NeuralNetworkMapping(input_dim, output_dim, n_hidden_layers=n_layers)
    elif mapping_type == 'decision_tree':
        # Paper doesn't specify max depth, using up to 8 for more complex rule-based dependencies
        max_depth = random.randint(2, 8)
        return DecisionTreeMapping(input_dim, output_dim, max_depth=max_depth)
    else:
        return CategoricalDiscretization(input_dim, output_dim)


# ============================================================================
# Initialization Data Sampling
# ============================================================================

def sample_initialization_data(
    n_samples: int,
    n_dims: int,
    init_type: str,
    init_scale: float,
    prototype_fraction: float = 0.0,
    prototype_temperature: float = 1.0,
) -> np.ndarray:
    """
    Generate initialization data for root nodes.
    
    Three sampling mechanisms (from paper):
    1. Normal: ε ~ N(0, σ²I)
    2. Uniform: ε ~ U(-a, a)
    3. Mixed: randomly select normal or uniform per root node
    
    Non-independence via prototype mixing:
    - Sample M = ρ*n prototype vectors
    - For each sample, compute weights α and mix prototypes
    """
    # Base sampling
    if init_type == 'normal':
        data = np.random.randn(n_samples, n_dims) * init_scale
    elif init_type == 'uniform':
        data = np.random.uniform(-init_scale, init_scale, (n_samples, n_dims))
    else:  # mixed - Paper: "for each root node, we randomly select either normal or uniform"
        # Since this function is called per root node, we select one distribution for all dims
        if random.random() < 0.5:
            data = np.random.randn(n_samples, n_dims) * init_scale
        else:
            data = np.random.uniform(-init_scale, init_scale, (n_samples, n_dims))
    
    # Apply prototype-based non-independence if specified
    if prototype_fraction > 0:
        n_prototypes = max(1, int(prototype_fraction * n_samples))
        
        # Sample prototypes from the data itself
        prototype_indices = np.random.choice(n_samples, n_prototypes, replace=False)
        prototypes = data[prototype_indices].copy()
        
        # For each sample, compute mixing weights
        # Using softmax over random distances with temperature
        new_data = np.zeros_like(data)
        for i in range(n_samples):
            # Random weights via Dirichlet (controlled by temperature)
            alpha = np.ones(n_prototypes) * prototype_temperature
            weights = np.random.dirichlet(alpha)
            new_data[i] = weights @ prototypes
        
        data = new_data
    
    return data


# ============================================================================
# Post-Processing
# ============================================================================

def kumaraswamy_transform(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Apply Kumaraswamy distribution warping.
    
    The Kumaraswamy distribution introduces nonlinear distortions.
    CDF: F(x) = 1 - (1 - x^a)^b for x in [0, 1]
    
    We first normalize to [0, 1], apply transform, then rescale.
    """
    # Normalize to [0, 1]
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return x
    
    x_normalized = (x - x_min) / (x_max - x_min + 1e-8)
    x_normalized = np.clip(x_normalized, 1e-8, 1 - 1e-8)
    
    # Apply Kumaraswamy CDF
    x_transformed = 1 - (1 - x_normalized ** a) ** b
    
    # Rescale back to original range
    return x_transformed * (x_max - x_min) + x_min


def quantize_feature(x: np.ndarray, n_bins: int = None) -> np.ndarray:
    """
    Quantize a continuous feature into discrete buckets.
    
    Mimics binned/discretized features common in real datasets.
    Paper: K categories with minimum 2 classes.
    """
    if n_bins is None:
        n_bins = random.randint(2, 20)  # Paper: minimum 2 categories
    
    # Sample bin edges from the feature values
    unique_vals = np.unique(x)
    if len(unique_vals) <= n_bins:
        return x  # Already discrete enough
    
    # Use percentile-based binning
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(x, percentiles[1:-1])
    
    return np.digitize(x, bin_edges).astype(float)


def add_missing_values(
    X: np.ndarray,
    missing_prob: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Introduce missing values using Missing Completely At Random (MCAR).
    
    Each value is masked as missing with probability ρ_miss,
    independently of the data values.
    
    Missing values are filled with NaN (float('nan')) to be handled by the model.
    """
    missing_mask = np.random.rand(*X.shape) < missing_prob
    
    # Ensure at least some non-missing values per feature
    for j in range(X.shape[1]):
        if missing_mask[:, j].sum() > X.shape[0] * 0.8:
            # Keep at least 20% non-missing
            missing_idx = np.where(missing_mask[:, j])[0]
            n_to_keep = int(X.shape[0] * 0.2)
            keep_idx = np.random.choice(missing_idx, min(len(missing_idx), n_to_keep), replace=False)
            missing_mask[keep_idx, j] = False
    
    X_with_missing = X.copy()
    X_with_missing[missing_mask] = float('nan')  # Use NaN for missing values
    
    return X_with_missing, missing_mask


def apply_post_processing(
    X: np.ndarray,
    hp: SCMHyperparameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply post-processing to features.
    
    Returns:
        X: Processed features
        categorical_mask: Boolean mask for categorical features
        missing_mask: Boolean mask for missing values
    """
    n_samples, n_features = X.shape
    categorical_mask = np.zeros(n_features, dtype=bool)
    missing_mask = np.zeros((n_samples, n_features), dtype=bool)
    
    for j in range(n_features):
        # Kumaraswamy warping - Paper: applied to some datasets (20%), 50% of features within those
        if hp.apply_kumaraswamy and random.random() < 0.5:
            X[:, j] = kumaraswamy_transform(X[:, j], hp.kumaraswamy_a, hp.kumaraswamy_b)
        
        # Quantization
        if random.random() < hp.quantization_prob:
            X[:, j] = quantize_feature(X[:, j])
            categorical_mask[j] = True
    
    # Missing values (applied to all features together)
    if hp.missing_prob > 0:
        X, missing_mask = add_missing_values(X, hp.missing_prob)
    
    return X, categorical_mask, missing_mask


# ============================================================================
# SCM-Based Data Generator
# ============================================================================

class SCMDataGenerator:
    """
    Generate synthetic datasets using Structural Causal Models.
    
    Following the TabPFN paper, this generator:
    1. Samples high-level hyperparameters
    2. Constructs a DAG using preferential attachment
    3. Assigns computational edge mappings
    4. Propagates initialization data through the graph
    5. Extracts features and targets from intermediate representations
    6. Applies post-processing
    """
    
    def __init__(
        self,
        n_samples_range: Tuple[int, int] = (10, 100),
        n_features_range: Tuple[int, int] = (2, 20),
        n_classes_range: Tuple[int, int] = (2, 10),
        is_regression: bool = False,
    ):
        self.n_samples_range = n_samples_range
        self.n_features_range = n_features_range
        self.n_classes_range = n_classes_range
        self.is_regression = is_regression
    
    def generate(
        self,
        n_samples: int = None,
        n_features: int = None,
        n_classes: int = None,
        train_ratio: float = 0.7,
    ) -> SyntheticDataset:
        """
        Generate a single synthetic dataset.
        
        Args:
            n_samples: Number of samples (random if None)
            n_features: Number of features (random if None)
            n_classes: Number of classes (random if None, ignored for regression)
            train_ratio: Fraction of samples for training
        """
        # Sample hyperparameters
        hp = sample_hyperparameters(
            n_samples_range=self.n_samples_range,
            n_features_range=self.n_features_range,
            n_classes_range=self.n_classes_range,
            is_regression=self.is_regression,
        )
        
        # Override with provided values
        if n_samples is not None:
            hp.n_samples = n_samples
        if n_features is not None:
            hp.n_features = n_features
        if n_classes is not None and not self.is_regression:
            hp.n_classes = n_classes
        
        # Ensure enough nodes for features and target
        hp.n_nodes = max(hp.n_nodes, hp.n_features + 1)
        
        # Step 1: Sample DAG structure
        adj = sample_dag_with_subgraphs(hp.n_nodes, hp.redirection_prob, hp.n_subgraphs)
        
        # Step 2: Sample edge mappings
        edge_mappings = {}
        for i in range(hp.n_nodes):
            parents = np.where(adj[i] > 0)[0]
            if len(parents) > 0:
                input_dim = len(parents) * hp.node_dim
                edge_mappings[i] = sample_edge_mapping(input_dim, hp.node_dim)
        
        # Track categorical discretization nodes for potential targets
        categorical_nodes = []
        for i, mapping in edge_mappings.items():
            if isinstance(mapping, CategoricalDiscretization):
                categorical_nodes.append((i, mapping))
        
        # Step 3: Generate initialization data and propagate through DAG
        node_values = {}  # node_id -> (n_samples, node_dim)
        
        # Find root nodes (no parents)
        root_nodes = [i for i in range(hp.n_nodes) if adj[i].sum() == 0]
        
        # Initialize root nodes
        for node in root_nodes:
            node_values[node] = sample_initialization_data(
                hp.n_samples, hp.node_dim,
                hp.init_type, hp.init_scale,
                hp.prototype_fraction, hp.prototype_temperature,
            )
        
        # Propagate through graph (topological order ensured by DAG structure)
        for i in range(hp.n_nodes):
            if i in node_values:
                continue  # Already initialized (root node)
            
            parents = np.where(adj[i] > 0)[0]
            if len(parents) == 0:
                # Root node not yet initialized
                node_values[i] = sample_initialization_data(
                    hp.n_samples, hp.node_dim,
                    hp.init_type, hp.init_scale,
                )
            else:
                # Concatenate parent values
                parent_values = np.concatenate([node_values[p] for p in parents], axis=1)
                
                # Apply edge mapping
                node_values[i] = edge_mappings[i](parent_values)
                
                # Add Gaussian noise
                node_values[i] += np.random.randn(*node_values[i].shape) * hp.edge_noise_std
        
        # Step 4: Sample feature and target node positions
        all_nodes = list(range(hp.n_nodes))
        
        # For classification: prefer categorical nodes for target
        if not self.is_regression and categorical_nodes:
            target_node_idx, target_mapping = random.choice(categorical_nodes)
            target_values = target_mapping.get_categories(
                np.concatenate([node_values[p] for p in np.where(adj[target_node_idx] > 0)[0]], axis=1)
                if adj[target_node_idx].sum() > 0 else node_values[target_node_idx]
            )
            hp.n_classes = target_mapping.n_categories
        else:
            # Select a random node for target
            target_node_idx = random.choice(all_nodes)
            # Use first dimension of node values as continuous target
            target_values = node_values[target_node_idx][:, 0]
            
            if not self.is_regression:
                # Convert continuous to discrete classes
                hp.n_classes = min(hp.n_classes, 10)
                if hp.n_classes == 2:
                    target_values = (target_values > np.median(target_values)).astype(np.int64)
                else:
                    percentiles = np.linspace(0, 100, hp.n_classes + 1)[1:-1]
                    thresholds = np.percentile(target_values, percentiles)
                    target_values = np.digitize(target_values, thresholds)
        
        # Select feature nodes (different from target)
        available_nodes = [n for n in all_nodes if n != target_node_idx]
        n_feature_nodes = min(hp.n_features, len(available_nodes))
        feature_nodes = random.sample(available_nodes, n_feature_nodes)
        
        # Step 5: Extract feature representations
        feature_dims_per_node = max(1, hp.n_features // n_feature_nodes)
        features = []
        
        for node in feature_nodes:
            node_data = node_values[node]
            # Extract some dimensions from this node
            n_dims = min(feature_dims_per_node, node_data.shape[1], hp.n_features - len(features))
            dim_indices = random.sample(range(node_data.shape[1]), n_dims)
            for d in dim_indices:
                features.append(node_data[:, d])
                if len(features) >= hp.n_features:
                    break
            if len(features) >= hp.n_features:
                break
        
        # Pad if needed
        while len(features) < hp.n_features:
            # Add noise features
            features.append(np.random.randn(hp.n_samples))
        
        X = np.stack(features[:hp.n_features], axis=1)
        
        # Paper: For regression, target should be a continuous feature WITHOUT post-processing
        # We extract the target before applying post-processing
        if self.is_regression:
            # Use target_values directly (continuous, no post-processing)
            y = target_values.astype(np.float32)
        else:
            y = target_values.astype(np.int64)
            
            # CRITICAL: Normalize labels to be consecutive 0, 1, ..., n_classes-1
            # This is essential for cross-entropy loss to work correctly.
            # TabPFN v1 does this in flexible_categorical.py with normalize_labels=True
            unique_labels = np.unique(y)
            if len(unique_labels) > 1:
                # Create mapping from original labels to 0, 1, 2, ...
                label_mapping = {old: new for new, old in enumerate(unique_labels)}
                y = np.array([label_mapping[label] for label in y], dtype=np.int64)
                hp.n_classes = len(unique_labels)
            else:
                # Only one class - this is a degenerate case, but handle gracefully
                y = np.zeros_like(y, dtype=np.int64)
                hp.n_classes = 1
        
        # Step 6: Apply post-processing to features (NOT to regression target)
        X, categorical_mask, missing_mask = apply_post_processing(X, hp)
        
        # Handle Inf values but preserve NaN for missing values (model handles NaN)
        X = np.where(np.isinf(X), np.sign(X) * 1e6, X)
        X = np.clip(np.where(np.isnan(X), X, np.clip(X, -1e6, 1e6)), -1e6, 1e6)
        
        # Compute train size
        train_size = max(1, int(hp.n_samples * train_ratio))
        train_size = min(train_size, hp.n_samples - 1)
        
        return SyntheticDataset(
            X=X.astype(np.float32),
            y=y,
            train_size=train_size,
            n_classes=0 if self.is_regression else hp.n_classes,
            is_regression=self.is_regression,
            categorical_mask=categorical_mask,
            missing_mask=missing_mask,
        )


# ============================================================================
# Dataset Batch Generator and Storage
# ============================================================================

def _generate_single_dataset_worker(args: Tuple) -> Dict:
    """
    Worker function for parallel dataset generation.
    Must be a top-level function for multiprocessing to work.
    
    Args:
        args: Tuple of (worker_id, seed, generator_params, max_samples, max_features)
    
    Returns:
        Dictionary with dataset arrays
    """
    worker_id, seed, generator_params, max_samples, max_features = args
    
    # Set unique seed for this worker
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    
    # Create generator with params
    generator = SCMDataGenerator(
        n_samples_range=generator_params['n_samples_range'],
        n_features_range=generator_params['n_features_range'],
        n_classes_range=generator_params['n_classes_range'],
        is_regression=generator_params['is_regression'],
    )
    
    # Generate dataset
    train_ratio = random.uniform(*generator_params['train_ratio_range'])
    dataset = generator.generate(train_ratio=train_ratio)
    
    n_samples = min(dataset.X.shape[0], max_samples)
    n_features = min(dataset.X.shape[1], max_features)
    
    # Prepare result arrays (padded to max sizes)
    X = np.zeros((max_samples, max_features), dtype=np.float32)
    y = np.zeros(max_samples, dtype=np.float32)
    categorical_mask = np.zeros(max_features, dtype=bool)
    missing_mask = np.zeros((max_samples, max_features), dtype=bool)
    
    X[:n_samples, :n_features] = dataset.X[:n_samples, :n_features]
    y[:n_samples] = dataset.y[:n_samples]
    
    if dataset.categorical_mask is not None:
        categorical_mask[:n_features] = dataset.categorical_mask[:n_features]
    if dataset.missing_mask is not None:
        missing_mask[:n_samples, :n_features] = dataset.missing_mask[:n_samples, :n_features]
    
    return {
        'X': X,
        'y': y,
        'train_size': min(dataset.train_size, n_samples - 1),
        'n_features': n_features,
        'n_samples': n_samples,
        'categorical_mask': categorical_mask,
        'missing_mask': missing_mask,
    }


def _generate_batch_parallel(
    batch_size: int,
    generator_params: Dict,
    max_samples: int,
    max_features: int,
    n_workers: int,
    base_seed: int,
) -> Dict[str, np.ndarray]:
    """
    Generate a batch of datasets in parallel using multiprocessing.
    
    Args:
        batch_size: Number of datasets to generate
        generator_params: Parameters for the SCMDataGenerator
        max_samples: Maximum samples per dataset
        max_features: Maximum features per dataset
        n_workers: Number of parallel workers
        base_seed: Base random seed (each worker gets base_seed + worker_id)
    
    Returns:
        Dictionary with batched arrays
    """
    # Prepare arguments for each worker
    args_list = [
        (i, base_seed, generator_params, max_samples, max_features)
        for i in range(batch_size)
    ]
    
    # Use multiprocessing pool
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_generate_single_dataset_worker, args_list)
    
    # Aggregate results
    X_batch = np.stack([r['X'] for r in results], axis=0)
    y_batch = np.stack([r['y'] for r in results], axis=0)
    train_sizes = np.array([r['train_size'] for r in results], dtype=np.int32)
    n_features_list = np.array([r['n_features'] for r in results], dtype=np.int32)
    n_samples_list = np.array([r['n_samples'] for r in results], dtype=np.int32)
    categorical_masks = np.stack([r['categorical_mask'] for r in results], axis=0)
    missing_masks = np.stack([r['missing_mask'] for r in results], axis=0)
    
    return {
        'X': X_batch,
        'y': y_batch,
        'train_size': train_sizes,
        'n_features': n_features_list,
        'n_samples': n_samples_list,
        'categorical_mask': categorical_masks,
        'missing_mask': missing_masks,
    }


class SyntheticDatasetGenerator:
    """
    Generate and save batches of synthetic datasets for training.
    Supports parallel generation using multiple CPU cores.
    """
    
    def __init__(
        self,
        n_samples_range: Tuple[int, int] = (10, 100),
        n_features_range: Tuple[int, int] = (2, 20),
        n_classes_range: Tuple[int, int] = (2, 10),
        train_ratio_range: Tuple[float, float] = (0.5, 0.9),
        is_regression: bool = False,
        n_workers: int = None,
    ):
        self.n_samples_range = n_samples_range
        self.n_features_range = n_features_range
        self.n_classes_range = n_classes_range
        self.train_ratio_range = train_ratio_range
        self.is_regression = is_regression
        
        # Set number of workers (default to number of CPU cores)
        if n_workers is None:
            self.n_workers = mp.cpu_count()
        else:
            self.n_workers = max(1, n_workers)
        
        # Keep a local generator for non-parallel use
        self.generator = SCMDataGenerator(
            n_samples_range=n_samples_range,
            n_features_range=n_features_range,
            n_classes_range=n_classes_range,
            is_regression=is_regression,
        )
        
        # Parameters dict for passing to workers
        self._generator_params = {
            'n_samples_range': n_samples_range,
            'n_features_range': n_features_range,
            'n_classes_range': n_classes_range,
            'train_ratio_range': train_ratio_range,
            'is_regression': is_regression,
        }
    
    def generate_batch(
        self,
        batch_size: int,
        max_samples: int,
        max_features: int,
        max_classes: int,
        parallel: bool = True,
        base_seed: int = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a batch of synthetic datasets.
        
        Args:
            batch_size: Number of datasets to generate
            max_samples: Maximum samples per dataset
            max_features: Maximum features per dataset
            max_classes: Maximum number of classes
            parallel: Whether to use parallel generation (default: True)
            base_seed: Base random seed for reproducibility
        
        Returns:
            Dictionary with:
            - X: (batch, max_samples, max_features) feature tensor
            - y: (batch, max_samples) target tensor
            - train_size: (batch,) train/test split positions
            - n_features: (batch,) actual number of features
            - n_samples: (batch,) actual number of samples
            - categorical_mask: (batch, max_features) categorical feature indicators
            - missing_mask: (batch, max_samples, max_features) missing value indicators
        """
        if base_seed is None:
            base_seed = random.randint(0, 2**31 - 1)
        
        # Use parallel generation if enabled and batch is large enough
        if parallel and self.n_workers > 1 and batch_size >= self.n_workers:
            return _generate_batch_parallel(
                batch_size=batch_size,
                generator_params=self._generator_params,
                max_samples=max_samples,
                max_features=max_features,
                n_workers=self.n_workers,
                base_seed=base_seed,
            )
        
        # Fall back to sequential generation
        X_batch = np.zeros((batch_size, max_samples, max_features), dtype=np.float32)
        y_batch = np.zeros((batch_size, max_samples), dtype=np.float32)
        train_sizes = np.zeros(batch_size, dtype=np.int32)
        n_features_list = np.zeros(batch_size, dtype=np.int32)
        n_samples_list = np.zeros(batch_size, dtype=np.int32)
        categorical_masks = np.zeros((batch_size, max_features), dtype=bool)
        missing_masks = np.zeros((batch_size, max_samples, max_features), dtype=bool)
        
        for i in range(batch_size):
            train_ratio = random.uniform(*self.train_ratio_range)
            
            # Generate dataset
            dataset = self.generator.generate(train_ratio=train_ratio)
            
            n_samples = min(dataset.X.shape[0], max_samples)
            n_features = min(dataset.X.shape[1], max_features)
            
            # Store in batch arrays (with padding)
            X_batch[i, :n_samples, :n_features] = dataset.X[:n_samples, :n_features]
            y_batch[i, :n_samples] = dataset.y[:n_samples]
            train_sizes[i] = min(dataset.train_size, n_samples - 1)
            n_features_list[i] = n_features
            n_samples_list[i] = n_samples
            
            if dataset.categorical_mask is not None:
                categorical_masks[i, :n_features] = dataset.categorical_mask[:n_features]
            if dataset.missing_mask is not None:
                missing_masks[i, :n_samples, :n_features] = dataset.missing_mask[:n_samples, :n_features]
        
        return {
            'X': X_batch,
            'y': y_batch,
            'train_size': train_sizes,
            'n_features': n_features_list,
            'n_samples': n_samples_list,
            'categorical_mask': categorical_masks,
            'missing_mask': missing_masks,
        }
    
    def generate_and_save(
        self,
        output_path: str,
        n_datasets: int,
        batch_size: int = 1000,
        max_samples: int = 100,
        max_features: int = 20,
        max_classes: int = 10,
        parallel: bool = True,
    ):
        """
        Generate datasets and save to HDF5 file.
        
        Args:
            output_path: Path to output HDF5 file
            n_datasets: Total number of datasets to generate
            batch_size: Number of datasets per batch
            max_samples: Maximum samples per dataset
            max_features: Maximum features per dataset
            max_classes: Maximum number of classes
            parallel: Whether to use parallel generation (default: True)
        """
        n_batches = (n_datasets + batch_size - 1) // batch_size
        
        if parallel and self.n_workers > 1:
            print(f"Using {self.n_workers} parallel workers for data generation")
        
        with h5py.File(output_path, 'w') as f:
            # Create datasets with chunked storage
            f.create_dataset('X', shape=(0, max_samples, max_features),
                           maxshape=(None, max_samples, max_features),
                           chunks=(min(batch_size, 100), max_samples, max_features),
                           dtype='float32', compression='lzf')
            f.create_dataset('y', shape=(0, max_samples),
                           maxshape=(None, max_samples),
                           chunks=(min(batch_size, 100), max_samples),
                           dtype='float32')
            f.create_dataset('single_eval_pos', shape=(0,),
                           maxshape=(None,), chunks=(min(batch_size, 100),), dtype='int32')
            f.create_dataset('num_features', shape=(0,),
                           maxshape=(None,), chunks=(min(batch_size, 100),), dtype='int32')
            f.create_dataset('num_datapoints', shape=(0,),
                           maxshape=(None,), chunks=(min(batch_size, 100),), dtype='int32')
            f.create_dataset('categorical_mask', shape=(0, max_features),
                           maxshape=(None, max_features),
                           chunks=(min(batch_size, 100), max_features), dtype='bool')
            f.create_dataset('missing_mask', shape=(0, max_samples, max_features),
                           maxshape=(None, max_samples, max_features),
                           chunks=(min(batch_size, 100), max_samples, max_features), dtype='bool')
            
            # Metadata
            f.create_dataset('max_num_classes', data=np.array([max_classes]))
            f.create_dataset('problem_type', data='regression' if self.is_regression else 'classification')
            
            # Generate batches
            total_generated = 0
            base_seed = random.randint(0, 2**31 - 1)
            pbar = tqdm(range(n_batches), desc="Generating datasets")
            
            for batch_idx in pbar:
                current_batch_size = min(batch_size, n_datasets - total_generated)
                if current_batch_size <= 0:
                    break
                
                # Use different seed for each batch to ensure reproducibility
                batch_seed = base_seed + batch_idx * batch_size
                
                batch = self.generate_batch(
                    current_batch_size, max_samples, max_features, max_classes,
                    parallel=parallel, base_seed=batch_seed
                )
                
                # Append to HDF5
                n = f['X'].shape[0]
                for key in ['X', 'y', 'categorical_mask', 'missing_mask']:
                    f[key].resize(n + current_batch_size, axis=0)
                    f[key][n:n+current_batch_size] = batch[key]
                
                f['single_eval_pos'].resize(n + current_batch_size, axis=0)
                f['single_eval_pos'][n:n+current_batch_size] = batch['train_size']
                
                f['num_features'].resize(n + current_batch_size, axis=0)
                f['num_features'][n:n+current_batch_size] = batch['n_features']
                
                f['num_datapoints'].resize(n + current_batch_size, axis=0)
                f['num_datapoints'][n:n+current_batch_size] = batch['n_samples']
                
                total_generated += current_batch_size
                pbar.set_postfix({'total': total_generated})
        
        print(f"Generated {total_generated} datasets, saved to {output_path}")


# ============================================================================
# Visualization
# ============================================================================

def visualize_datasets(n_datasets: int = 6, n_samples: int = 200, n_features: int = 2):
    """Visualize samples from the SCM generator (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return
    
    generator = SCMDataGenerator(
        n_samples_range=(n_samples, n_samples),
        n_features_range=(n_features, n_features),
        n_classes_range=(2, 5),
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    for i, ax in enumerate(axes.flat):
        dataset = generator.generate()
        
        n_classes = dataset.n_classes
        if n_features >= 2:
            for c in range(n_classes):
                mask = dataset.y == c
                ax.scatter(dataset.X[mask, 0], dataset.X[mask, 1], 
                          alpha=0.6, label=f'Class {c}', s=20)
        else:
            for c in range(n_classes):
                mask = dataset.y == c
                ax.scatter(dataset.X[mask, 0], np.zeros_like(dataset.X[mask, 0]) + c*0.1,
                          alpha=0.6, label=f'Class {c}', s=20)
        
        ax.set_title(f'Dataset {i+1} ({n_classes} classes)')
        ax.legend(fontsize=8)
    
    plt.suptitle('SCM-Based Synthetic Datasets', fontsize=14)
    plt.tight_layout()
    plt.savefig('scm_samples.png', dpi=150)
    plt.show()
    print("Saved visualization to scm_samples.png")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic tabular data for TabPFN training using SCMs'
    )
    
    parser.add_argument('--output', '-o', type=str, default='data/synthetic.h5',
                       help='Output HDF5 file path')
    parser.add_argument('--n_datasets', '-n', type=int, default=100000,
                       help='Number of datasets to generate')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Batch size for generation')
    parser.add_argument('--max_samples', type=int, default=512,
                       help='Maximum samples per dataset (paper uses up to 2048)')
    parser.add_argument('--max_features', type=int, default=160,
                       help='Maximum features per dataset (paper uses Beta distribution scaled to 1-160)')
    parser.add_argument('--max_classes', type=int, default=10,
                       help='Maximum number of classes')
    parser.add_argument('--regression', action='store_true',
                       help='Generate regression datasets instead of classification')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers (default: all CPU cores)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel generation')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize sample datasets instead of generating data')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.visualize:
        visualize_datasets()
        return
    
    # Generate data
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    generator = SyntheticDatasetGenerator(
        n_samples_range=(32, args.max_samples),
        n_features_range=(2, args.max_features),
        n_classes_range=(2, args.max_classes),
        is_regression=args.regression,
        n_workers=args.workers,
    )
    
    generator.generate_and_save(
        output_path=args.output,
        n_datasets=args.n_datasets,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_features=args.max_features,
        max_classes=args.max_classes,
        parallel=not args.no_parallel,
    )


if __name__ == '__main__':
    import sys
    
    # If no arguments provided, run tests
    if len(sys.argv) == 1:
        print("Testing SyntheticDatasetGenerator...")
        
        # Set seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Create generator
        generator = SyntheticDatasetGenerator(
            n_samples_range=(20, 50),
            n_features_range=(5, 10),
            n_classes_range=(2, 5),
        )
        
        # Test hyperparameter sampling
        params = sample_hyperparameters(
            n_samples_range=(20, 50),
            n_features_range=(5, 10),
            n_classes_range=(2, 5),
        )
        print(f"Sampled hyperparameters: n_nodes={params.n_nodes}, n_features={params.n_features}, "
              f"n_samples={params.n_samples}, n_classes={params.n_classes}")
        
        # Test single dataset generation (using internal SCMDataGenerator)
        print("\nGenerating single dataset...")
        dataset = generator.generator.generate()
        print(f"  X shape: {dataset.X.shape}")
        print(f"  y shape: {dataset.y.shape}")
        print(f"  train_size: {dataset.train_size}")
        print(f"  n_classes: {dataset.n_classes}")
        print(f"  y unique values: {np.unique(dataset.y)}")
        
        # Test batch generation
        print("\nGenerating batch of 5 datasets...")
        batch = generator.generate_batch(5, max_samples=50, max_features=10, max_classes=5)
        print(f"  X batch shape: {batch['X'].shape}")
        print(f"  y batch shape: {batch['y'].shape}")
        print(f"  train_sizes: {batch['train_size']}")
        print(f"  n_classes: {batch['n_features']}")
        # Verify data is valid
        assert not np.isnan(batch['X']).all(), "X contains all NaN values"
        assert batch['X'].shape[0] == 5, "Batch size mismatch"
        
        print("\n✓ Data generation test passed!")
    else:
        main()

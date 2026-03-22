# OpenTab Improvement Program

## 1. Setup

At the start of each improvement session:

1. Create branch `autoimprove/<tag>` (tag = today's date, e.g. `mar22`)
2. Read these files to understand the codebase:
   - `model.py` — Transformer architecture (encoder, dual attention, decoder) + sklearn wrappers
   - `train.py` — Training loop (online/offline data, gradient accumulation, mixed precision)
   - `generate_data.py` — SCM-based synthetic data generation (DAG sampling, edge mappings)
   - `evaluate.py` — Evaluation: sklearn quick-eval + TabArena benchmark
   - `.autoimprove/analysis.md` — Repository analysis and quality issues
3. Read `.autoimprove/results.tsv` to see previous experiment results
4. Verify baseline exists: `cat .autoimprove/baselines/baseline.json`
5. Begin the experiment loop

## 2. Scope

### What you CAN modify

- **`model.py`** — Transformer architecture, attention mechanisms, encoder/decoder design, sklearn wrappers (`OpenTabClassifier`, `OpenTabRegressor`). Fair game: layer design, initialization, normalization, activation functions, inference batching, caching.
- **`train.py`** — Training loop, loss computation, optimizer/scheduler, data loading, gradient accumulation, mixed precision. Fair game: loss functions, learning rate schedules, data augmentation, training efficiency.
- **`generate_data.py`** — Synthetic data generation pipeline, SCM structure, edge mappings, post-processing. Fair game: activation functions, graph sampling, initialization schemes, data quality improvements.
- **`evaluate.py`** — Evaluation wrappers and TabArena integration. Fair game: prediction calibration, ensemble strategies, preprocessing.

### What you CANNOT modify

- **`tests/`** — Test suite is ground truth for correctness. All 4 tests must pass.
- **`.autoimprove/`** — Evaluation system. Modifying it invalidates all scores.
- **`pyproject.toml`** — Build config and dependencies.
- **`uv.lock`** — Dependency lockfile.
- **`__init__.py`** — Package marker.

## 3. Evaluation

### Command

```bash
uv run .autoimprove/eval_harness.py
```

### What it does

The harness runs one evaluator:

| Evaluator | Weight | What it measures |
|---|---|---|
| `train_and_eval` | 4.0 | Train for 240s, measure mean accuracy on Iris/Wine/Breast Cancer |

### Output format

```json
{
  "composite_score": 0.44191,
  "elapsed_seconds": 246.5,
  "evaluators": [
    {
      "name": "train_and_eval",
      "score": 0.44191,
      "weight": 4.0,
      "details": {
        "mean_accuracy": 0.44191,
        "per_dataset": {"Iris": 0.3111, "Wine": 0.3889, "Breast Cancer": 0.6257},
        "training_steps": 1350,
        "final_loss": 1.2167,
        "elapsed_seconds": 243.9,
        "device": "cuda"
      }
    }
  ]
}
```

### Primary metric

The `train_and_eval` evaluator is the primary (and only) metric. It trains a small model (embedding_size=64, n_layers=3) for 240 seconds using online data generation, then measures classification accuracy on 3 sklearn datasets. The score is the mean accuracy — higher is better.

### Time budget

Each evaluation run completes within 300 seconds total (240s training + ~60s evaluation).

### Quick score extraction

```bash
uv run .autoimprove/eval_harness.py 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['composite_score'])"
```

### Baseline scores

| Evaluator | Baseline Score | Details |
|---|---|---|
| `train_and_eval` | **0.44191** | Iris=0.3111, Wine=0.3889, Breast Cancer=0.6257, 1350 steps, loss=1.2167 |

**Composite: 0.44191** (1350 training steps in 240s on CUDA)

## 4. The Experiment Loop

```
LOOP FOREVER:
1. Pick a focused improvement (one idea per experiment)
2. Edit the code (model.py, train.py, generate_data.py, or evaluate.py)
3. Run tests: uv run pytest tests/ -x
4. If tests fail: fix and retry
5. git commit -m "experiment: <description>"
6. Run: uv run .autoimprove/eval_harness.py
7. Read: cat .autoimprove/run.log (contains evaluator progress + final JSON)
8. Log to results.tsv
9. If improved: keep the commit, update baseline
10. If not improved: git reset --hard HEAD~1
11. GOTO 1

Time budget: 300 seconds per evaluation run.
Each experiment must be evaluable within this window.
```

## 5. Logging

Log every experiment to `.autoimprove/results.tsv` (tab-separated):

```
commit	composite_score	status	description
baseline	0.44191	keep	baseline — initial state
a1b2c3d	0.4821	keep	experiment: improve encoder initialization
d4e5f6g	0.4310	revert	experiment: add gradient checkpointing (no improvement)
h7i8j9k	0.0000	crash	experiment: new attention mechanism (NaN in loss)
```

## 6. Strategy

### Start with the primary metric

The score measures how well the model learns from 240s of training on synthetic data and generalizes to real classification tasks. Focus on changes that:

1. **Improve training efficiency** — More useful gradient steps in 240s. The current training loop processes batch items one at a time in `compute_loss`. Batching this would be a big speedup.
2. **Improve data generation quality** — Better synthetic data means the model learns more transferable features. The SCM pipeline has many knobs (graph structure, edge mappings, post-processing).
3. **Improve model architecture** — Better attention mechanisms, initialization, normalization. The current model is a standard pre-norm transformer with dual attention.
4. **Improve the loss function** — The current loss is plain cross-entropy. Label smoothing, focal loss, or other modifications could help generalization.
5. **Improve inference** — The sklearn wrappers handle the bridge between training and evaluation. Better calibration or preprocessing could improve accuracy.

### Simplicity criterion

All else equal, simpler is better. Removing code and getting equal or better results is a great outcome. A 0.001 accuracy improvement that adds 50 lines of complexity is probably not worth it.

### Crash recovery

If the experiment crashes:
- If it's a typo or trivial error, fix and re-run.
- If the idea is fundamentally broken (NaN loss, OOM), log the crash and move on.
- Check `cat .autoimprove/run.log` for the error details.

## 7. NEVER STOP

Once the loop begins, do NOT pause to ask the human if you should continue.
Do NOT ask "should I keep going?" or "is this a good stopping point?".
The human might be asleep. You are autonomous. If you run out of ideas,
think harder — re-read the source code, try combining previous experiments,
try more radical approaches. The loop runs until the human interrupts you.

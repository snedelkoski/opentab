# Contributing to OpenTab

Thank you for your interest in contributing to OpenTab! We welcome contributions from the community to help make this project better.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub. Include as much detail as possible:
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, etc.)

### Suggesting Enhancements

We welcome ideas for new features or improvements. Please open an issue to discuss your idea before submitting a Pull Request.

### Pull Requests

1. Fork the repository.
2. Create a new branch for your feature or fix: `git checkout -b feature/my-feature`.
3. Make your changes.
4. Run tests (if available) to ensure no regressions.
5. Commit your changes with clear messages.
6. Push to your fork and submit a Pull Request.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/snedelkoski/opentab.git
   cd opentab
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Coding Standards

- Follow PEP 8 style guidelines.
- Use type hints where possible.
- Write docstrings for new functions and classes.
- Ensure code is formatted (we use `black` and `ruff`).

## License

By contributing, you agree that your contributions will be licensed under the Apache License, Version 2.0.

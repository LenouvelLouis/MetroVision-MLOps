# Contributing to MetroVision-MLOps

Thank you for your interest in contributing!

## Getting started

1. Fork and clone the repository
2. Create a virtual environment with Python 3.11+
3. Install dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development workflow

- Create a feature branch from `main`
- Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages
  (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `ci:`)
- Write tests for every new module (pytest)
- Run the linter before pushing:
  ```bash
  ruff check . && ruff format --check .
  ```
- Open a pull request against `main`

## Code standards

- Python 3.11+, type hints everywhere
- Google-style docstrings
- Line length: 100 characters
- No `print()` in production code — use `logging`

## Questions?

Open an issue on GitHub.

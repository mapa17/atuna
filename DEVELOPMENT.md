# Development Guide for Atuna

This guide covers the complete development workflow using UV for the Atuna project.

## Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) package manager
- Git

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mapa17/atuna.git
cd atuna

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install development dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install

# Test the installation
uv run atuna --version
```

## Development Workflow

### 1. Environment Management

```bash
# Install Python 3.12 with UV
uv python install 3.12

# Create and sync environment
uv sync --dev --all-extras

# Install in editable mode
uv pip install -e .

# Show installed packages
uv pip list

# Show dependency tree
uv tree
```

### 2. Code Quality

```bash
# Run all checks
make ci

# Individual checks
uv run ruff check src/          # Linting
uv run ruff format src/         # Formatting
uv run ty check src/            # Type checking
uv run pre-commit run --all-files  # All pre-commit hooks
```

### 3. Testing

```bash
# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=atuna --cov-report=html

# Test CLI
uv run atuna --version
uv run atuna --list-models
```

### 4. Building and Publishing

```bash
# Build package
uv build

# Check package
uv run twine check dist/*

# Publish to TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/

# Publish to PyPI (for releases)
uv publish
```

## UV-Specific Features

### Dependency Management

```bash
# Add a dependency
uv add numpy

# Add a development dependency
uv add --dev pytest-asyncio

# Add an optional dependency
uv add --optional docs mkdocs

# Remove a dependency
uv remove numpy

# Update all dependencies
uv lock --upgrade
```

### Virtual Environment

```bash
# UV automatically manages virtual environments
# The environment is stored in .venv/

# Activate manually (usually not needed)
source .venv/bin/activate

# Run commands in the environment
uv run python script.py
uv run atuna --version
```

### Cross-Platform Scripts

```bash
# Install tools temporarily
uv run --with black black src/
uv run --with mypy mypy src/

# Run scripts from pyproject.toml
uv run --with mkdocs mkdocs serve
```

## GitHub Actions Integration

The project uses UV in GitHub Actions for:

1. **Fast dependency installation** - UV is significantly faster than pip
2. **Lockfile-based reproducible builds** - `uv.lock` ensures exact versions
3. **Efficient caching** - UV's cache is optimized for CI
4. **Cross-platform compatibility** - Works on Linux, macOS, and Windows

### Example Workflow Steps

```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v3
  with:
    enable-cache: true
    cache-dependency-glob: "uv.lock"

- name: Set up Python
  run: uv python install 3.12

- name: Install dependencies
  run: uv sync --dev

- name: Run tests
  run: uv run pytest
```

## Performance Benefits

UV provides significant performance improvements:

- **10-100x faster** dependency resolution
- **Parallel downloads** and installations
- **Efficient caching** across projects
- **Minimal disk I/O** with optimized storage

## Migration from pip/Poetry

If migrating from other tools:

```bash
# From requirements.txt
uv add --requirements requirements.txt

# From poetry
# Copy dependencies from pyproject.toml [tool.poetry.dependencies]
# to [project.dependencies] and run:
uv sync

# From pipenv
# Export Pipfile.lock and import:
pipenv requirements > requirements.txt
uv add --requirements requirements.txt
```

## Troubleshooting

### Common Issues

1. **UV not found**: Install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. **Cache issues**: Clear with `uv cache clean`
3. **Lock conflicts**: Update with `uv lock --upgrade`
4. **Python version**: Install with `uv python install 3.12`

### Debug Commands

```bash
# Show UV configuration
uv --version
uv python list

# Show project info
uv tree
uv pip list

# Clear cache if needed
uv cache clean
```

## Best Practices

1. **Always use `uv run`** for running commands in the project environment
2. **Commit `uv.lock`** for reproducible builds
3. **Use dependency groups** for organizing optional dependencies
4. **Leverage UV's caching** in CI by using the setup-uv action
5. **Pin Python version** in pyproject.toml for consistency
6. **Use `--dev` flag** for development installs

## Scripts and Shortcuts

The project includes a Makefile with common commands:

```bash
make help           # Show all available commands
make dev-install    # Setup development environment
make ci             # Run all CI checks
make build          # Build the package
make clean          # Clean build artifacts
```

## Monitoring and Debugging

```bash
# Start TensorBoard for training monitoring
make tensorboard

# Start Optuna dashboard for hyperparameter optimization
make optuna-dashboard

# Test examples
make example-basic
make example-hyper
```

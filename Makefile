# Makefile for Atuna development with UV

.PHONY: help install dev-install test lint format type-check build clean publish-test publish

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv sync --no-dev

dev-install: ## Install development dependencies
	uv sync --all-extras --dev
	uv run pre-commit install

test: ## Run tests
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage
	uv run pytest tests/ -v --cov=atuna --cov-report=html --cov-report=term

lint: ## Run linting
	uv run ruff check src/ examples/ tests/

lint-fix: ## Run linting with auto-fix
	uv run ruff check --fix src/ examples/ tests/

format: ## Run code formatting
	uv run ruff format src/ examples/ tests/

format-check: ## Check code formatting
	uv run ruff format --check src/ examples/ tests/

type-check: ## Run type checking
	uv run ty check src/ --ignore-missing-imports

pre-commit: ## Run pre-commit hooks
	uv run pre-commit run --all-files

ci: lint format-check type-check test ## Run all CI checks

build: ## Build the package
	uv build

build-check: build ## Build and check package
	uv run --with twine twine check dist/*

clean: ## Clean build artifacts
	rm -rf dist/
	rm -rf src/atuna.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

publish-test: build-check ## Publish to TestPyPI
	uv publish --publish-url https://test.pypi.org/legacy/

publish: build-check ## Publish to PyPI
	uv publish

install-test: ## Install from TestPyPI for testing
	uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ atuna

cli-test: ## Test CLI functionality
	uv run atuna --version

example-basic: ## Run basic example
	uv run python examples/basic_finetuning.py

example-hyper: ## Run hyperparameter example
	uv run python examples/hyperparameter_search.py

# Development server commands
tensorboard: ## Start TensorBoard server
	uv run tensorboard --logdir ./atuna_workspace/logs --host 0.0.0.0 --port 6006

optuna-dashboard: ## Start Optuna dashboard
	uv run optuna-dashboard sqlite:///./atuna_workspace/optuna_studies.db --host 0.0.0.0 --port 8080

# Environment management
python-install: ## Install Python versions with uv
	uv python install 3.12

python-list: ## List available Python versions
	uv python list

lock: ## Update lockfile
	uv lock

lock-upgrade: ## Upgrade and update lockfile
	uv lock --upgrade

tree: ## Show dependency tree
	uv tree

outdated: ## Show outdated packages
	uv tree --outdated

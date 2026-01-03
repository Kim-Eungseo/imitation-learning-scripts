.PHONY: setup format lint check train clean clean-env update-env info help

# Default target
help:
	@echo "🎯 Imitation Learning Scripts - Available Commands"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup        - Create conda environment and setup pre-commit"
	@echo "    make update-env   - Update conda environment"
	@echo ""
	@echo "  Development:"
	@echo "    make train        - Start training"
	@echo "    make format       - Format code with black and isort"
	@echo "    make lint         - Run flake8 linter"
	@echo "    make check        - Format and lint"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean        - Clean output files"
	@echo "    make clean-env    - Remove conda environment"
	@echo ""
	@echo "  Info:"
	@echo "    make info         - Show environment information"

# Conda environment setup
setup:
	@echo "🐍 Creating conda environment..."
	@conda env create -f environment.yml
	@echo ""
	@echo "✅ Environment created! Next steps:"
	@echo "   1. conda activate imitation-learning"
	@echo "   2. pre-commit install  (optional, for git hooks)"

# Update conda environment
update-env:
	@echo "🔄 Updating conda environment..."
	@conda env update -f environment.yml --prune
	@echo "✅ Environment updated!"

# Code formatting
format:
	@echo "🎨 Formatting code with black..."
	@black --line-length 120 .
	@echo "📦 Sorting imports with isort..."
	@isort --profile black --line-length 120 .
	@echo "✅ Formatting complete!"

# Linting
lint:
	@echo "🔍 Running flake8..."
	@flake8 .
	@echo "✅ Linting complete!"

# Check code quality (format + lint)
check: format lint

# Train model
train:
	@echo "🚀 Starting training..."
	@python train.py

# Clean outputs
clean:
	@echo "🧹 Cleaning outputs..."
	@rm -rf outputs/
	@rm -rf __pycache__/
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✨ Clean complete!"

# Clean conda environment
clean-env:
	@echo "🗑️  Removing conda environment..."
	@conda env remove -n imitation-learning
	@echo "✅ Environment removed!"

# Show environment info
info:
	@echo "📊 Environment Information"
	@echo ""
	@echo "Conda environments:"
	@conda env list
	@echo ""
	@echo "Current environment:"
	@conda info --envs | grep '*' || echo "  No conda environment activated"
	@echo ""
	@which python 2>/dev/null && python --version || echo "Python not found"
	@which conda 2>/dev/null && conda --version || echo "Conda not found"

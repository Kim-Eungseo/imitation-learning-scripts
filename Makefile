.PHONY: setup format lint check train train-diffusion train-act train-tdmpc clean clean-env update-env info help

# Default target
help:
	@echo "🎯 Imitation Learning Scripts - Available Commands"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup        - Create conda environment and setup pre-commit"
	@echo "    make update-env   - Update conda environment"
	@echo ""
	@echo "  Training:"
	@echo "    make train        - Train with Diffusion Policy (default)"
	@echo "    make train-diffusion - Train with Diffusion Policy"
	@echo "    make train-act    - Train with ACT (Action Chunking Transformer)"
	@echo "    make train-tdmpc  - Train with TDMPC"
	@echo ""
	@echo "  Development:"
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
	@echo ""
	@echo "  Examples:"
	@echo "    python train.py --policy act --training_steps 10000"
	@echo "    python train.py --policy tdmpc --batch_size 32"
	@echo "    python train.py --help  # Show all options"

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

# Train with Diffusion Policy (default)
train: train-diffusion

train-diffusion:
	@echo "🚀 Starting Diffusion Policy training..."
	@python train.py --policy diffusion

# Train with ACT
train-act:
	@echo "🚀 Starting ACT training..."
	@python train.py --policy act

# Train with TDMPC
train-tdmpc:
	@echo "🚀 Starting TDMPC training..."
	@python train.py --policy tdmpc

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
	@echo ""
	@echo "Supported policies: diffusion, act, tdmpc"

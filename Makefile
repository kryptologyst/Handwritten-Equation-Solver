# Makefile for Handwritten Equation Solver

.PHONY: help install test lint format clean run docker-build docker-run generate-samples

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

test: ## Run tests
	pytest test_solver.py -v

test-cov: ## Run tests with coverage
	pytest test_solver.py -v --cov=modern_solver --cov-report=html --cov-report=term

lint: ## Run linting
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

type-check: ## Run type checking
	mypy modern_solver.py --ignore-missing-imports

format: ## Format code with black
	black .

format-check: ## Check code formatting
	black --check .

run: ## Run the Streamlit web application
	streamlit run modern_solver.py

run-cli: ## Run CLI with sample image
	python cli_solver.py sample_images/equation_01.png --verbose

generate-samples: ## Generate sample equation images
	python generate_samples.py

clean: ## Clean up generated files
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

docker-build: ## Build Docker image
	docker build -t handwritten-equation-solver .

docker-run: ## Run Docker container
	docker run -p 8501:8501 handwritten-equation-solver

docker-compose-up: ## Start with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop docker-compose
	docker-compose down

setup: install-dev generate-samples ## Complete setup for development
	@echo "Setup complete! Run 'make run' to start the application."

all-checks: lint type-check format-check test ## Run all checks
	@echo "All checks passed!"

# Default target
.DEFAULT_GOAL := help

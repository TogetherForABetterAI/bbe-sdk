.PHONY: help install install-dev test test-cov lint format clean build

help:
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=bbe_sdk --cov-report=html --cov-report=term

lint:
	flake8 bbe_sdk/ tests/
	mypy bbe_sdk/

format:
	black bbe_sdk/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python -m build

install-from-github:
	@if [ -z "$(GITHUB_TOKEN)" ]; then echo "Error: GITHUB_TOKEN not set"; exit 1; fi
	@if [ -z "$(GITHUB_ORG)" ]; then echo "Error: GITHUB_ORG not set"; exit 1; fi
	pip install "git+https://$(GITHUB_TOKEN)@github.com/$(GITHUB_ORG)/bbe-sdk.git@$(VERSION)"

check: lint test

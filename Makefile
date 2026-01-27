.PHONY: help install test clean docker-build docker-up docker-down docs train export benchmark

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        - Install Python dependencies"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make clean          - Clean generated files"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-up      - Start Docker containers"
	@echo "  make docker-down    - Stop Docker containers"
	@echo "  make docker-logs    - View Docker logs"
	@echo "  make docs           - Generate documentation"
	@echo "  make train          - Train EEG model"
	@echo "  make export         - Export models"
	@echo "  make benchmark      - Benchmark models"

# Install dependencies
install:
	pip install -r requirements.txt

# Run all tests
test:
	pytest -v

# Run unit tests
test-unit:
	pytest -v -m unit

# Run integration tests
test-integration:
	pytest -v -m integration

# Clean generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf docs/html/

# Docker commands
docker-build:
	docker-compose build --no-cache

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f edge-models

docker-restart: docker-down docker-up

# Production Docker
docker-prod-up:
	docker-compose -f docker-compose.prod.yml up -d

docker-prod-down:
	docker-compose -f docker-compose.prod.yml down

# Generate documentation
docs:
	python scripts/generate_docs.py

# Training
train:
	python scripts/train_eeg_model.py --epochs 100 --batch-size 32

# Export models
export:
	python scripts/export_model.py --model-type tenn_eeg --format all

# Benchmark
benchmark:
	python scripts/benchmark.py --model-type tenn_eeg --iterations 1000

# Run FastAPI locally
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Format code (requires black)
format:
	black app/ scripts/

# Lint code (requires pylint)
lint:
	pylint app/ scripts/

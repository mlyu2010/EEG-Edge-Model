# Setup Guide - EEG Edge Model

Complete setup instructions for the EEG Edge Model project implementing TENN models for BrainChip's Akida hardware.

## Prerequisites

1. **Docker & Docker Compose**
   - Docker 20.10+
   - Docker Compose 2.0+

2. **Python 3.12+**
   - Python 3.12 or higher
   - pip package manager
   - Virtual environment support

3. **Hardware Acceleration (Optional)**
   - **CUDA**: NVIDIA GPU with CUDA 11.8+ for GPU acceleration
   - **MPS**: Apple Silicon (M1/M2/M3) for GPU acceleration on macOS
   - **CPU**: Works on all platforms without GPU

## Quick Start (Docker)

### 1. Clone and Setup

```bash
cd /Users/mlyu2010/IdeaProjects/EEG-Edge-Model
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings if needed
```

### 3. Build and Run

```bash
# Using Makefile
make docker-build
make docker-up
make docker-logs

# Or using docker-compose directly
docker-compose down
docker-compose build --no-cache
docker-compose up -d
docker-compose logs -f edge-models
```

### 4. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Local Development Setup

### 1. Create Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**For NVIDIA GPU Support:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For Apple Silicon (MPS):**
PyTorch MPS support is included by default in PyTorch 2.0+.

### 3. Run Locally

```bash
# Using Makefile
make run

# Or directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Project Structure Setup

The project structure is already created with these key directories:

```
app/          - Main application code
├── api/      - FastAPI routes
├── core/     - Configuration
├── models/   - Neural network models
├── utils/    - Utility functions
└── tests/    - Test suite

configs/      - Configuration files
data/         - Data storage (raw/processed)
models/       - Model storage (trained/quantized/exported)
scripts/      - Utility scripts
logs/         - Application logs
```

## Development Workflow

### 1. Training Models

Train the TENN EEG model:

```bash
# Auto-detect best device (recommended)
python scripts/train_eeg_model.py \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --device auto

# Apple Silicon GPU (M1/M2/M3 Macs)
python scripts/train_eeg_model.py \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --device mps

# NVIDIA GPU
python scripts/train_eeg_model.py \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --device cuda

# CPU only
python scripts/train_eeg_model.py \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --device cpu
```

**Device Selection Tips:**
- Use `auto` for automatic device detection
- MPS provides ~2-5x speedup on Apple Silicon
- CUDA provides ~10-100x speedup on NVIDIA GPUs
- CPU works everywhere but is slowest

### 2. Exporting Models

Export to multiple formats:

```bash
# Export to ONNX
python scripts/export_model.py \
    --model-type tenn_eeg \
    --format onnx \
    --verify

# Export to TorchScript
python scripts/export_model.py \
    --model-type tenn_eeg \
    --format torchscript

# Export for Akida (requires Akida SDK)
python scripts/export_model.py \
    --model-type tenn_eeg \
    --format akida

# Export all formats
python scripts/export_model.py \
    --model-type tenn_eeg \
    --format all
```

### 3. Benchmarking

Compare performance across frameworks and devices:

```bash
# Benchmark on auto-detected device
python scripts/benchmark.py \
    --model-type tenn_eeg \
    --iterations 1000 \
    --device auto \
    --frameworks pytorch tvm onnx

# Benchmark on Apple Silicon
python scripts/benchmark.py \
    --model-type tenn_eeg \
    --iterations 1000 \
    --device mps

# Benchmark on NVIDIA GPU
python scripts/benchmark.py \
    --model-type tenn_eeg \
    --iterations 1000 \
    --device cuda

# Compare CPU vs GPU performance
python scripts/benchmark.py \
    --model-type tenn_eeg \
    --iterations 1000 \
    --device cpu
```

### 4. Generate Documentation

```bash
python scripts/generate_docs.py
# View at: docs/html/app/index.html
```

## Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# With coverage report
pytest --cov=app --cov-report=html
```

### Run Specific Test Files

```bash
pytest app/tests/unit/test_models.py
pytest app/tests/integration/test_api.py
```

## Using the API

### List Available Models

```bash
curl http://localhost:8000/api/v1/models
```

### EEG Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict/eeg \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[0.1, 0.2, ...], ...],
    "model_type": "eeg"
  }'
```

### Quantize Model

```bash
curl -X POST "http://localhost:8000/api/v1/model/tenn_eeg_v1/quantize?target=akida"
```

### Benchmark Model

```bash
curl http://localhost:8000/api/v1/benchmark/tenn_eeg_v1?iterations=100
```

## Docker Production Deployment

### 1. Build Production Image

```bash
docker-compose -f docker-compose.prod.yml build
```

### 2. Run Production

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Monitor Logs

```bash
docker-compose -f docker-compose.prod.yml logs -f
```

## Akida SDK Integration

To integrate with actual BrainChip Akida hardware:

### 1. Install Akida SDK

```bash
# Contact BrainChip for SDK access
pip install akida
```

### 2. Update Quantization Code

Edit `app/utils/akida_quantization.py` to uncomment Akida SDK calls.

### 3. Configure Device

```bash
# In .env file
AKIDA_DEVICE_ID=0
AKIDA_NUM_DEVICES=1
```

### 4. Quantize and Deploy

```bash
python scripts/export_model.py \
    --model-type tenn_eeg \
    --format akida
```

## TVM Compilation

### 1. Compile for Target

```bash
python scripts/benchmark.py \
    --model-type tenn_eeg \
    --frameworks tvm
```

### 2. Multi-Target Compilation

Edit `configs/model_config.yaml` to specify targets:

```yaml
tvm:
  targets:
    - "llvm"
    - "llvm -mcpu=core-avx2"
    - "llvm -mcpu=native"
```

### 3. Auto-Tuning (Advanced)

Uncomment auto-tuning code in `app/utils/tvm_compiler.py`.

## Common Operations with Makefile

The project includes a Makefile for convenience:

```bash
make help              # Show all available commands
make install           # Install dependencies
make test              # Run tests
make docker-up         # Start Docker
make docker-down       # Stop Docker
make docs              # Generate documentation
make train             # Train model
make export            # Export models
make benchmark         # Benchmark models
make clean             # Clean generated files
```

## Troubleshooting

### Docker Issues

**Problem**: Container fails to start

```bash
# Check logs
docker-compose logs edge-models

# Rebuild without cache
docker-compose build --no-cache
```

**Problem**: Port 8000 already in use

```bash
# Change port in docker-compose.yml
ports:
  - "8080:8000"  # Use port 8080 instead
```

### Python Issues

**Problem**: Import errors

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Problem**: CUDA not available

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Problem**: MPS not available on Apple Silicon

```bash
# Ensure you have macOS 12.3+ and PyTorch 2.0+
pip install --upgrade torch torchvision

# Check device availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Problem**: Want to check available devices

```bash
# Check all available devices
python -c "from app.utils.device import print_device_info; print_device_info()"
```

### TVM Issues

**Problem**: TVM compilation fails

```bash
# TVM is optional - models work without it
# For TVM support, ensure LLVM is installed
brew install llvm  # macOS
apt-get install llvm  # Ubuntu
```

## Environment Variables

Key environment variables in `.env`:

```bash
# Application
APP_NAME=EEG-Edge-Model
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Directories
MODEL_DIR=./models
DATA_DIR=./data
LOG_DIR=./logs

# Training
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=0.001

# Device (auto, cpu, cuda, mps)
DEVICE=auto

# Akida (if available)
AKIDA_DEVICE_ID=0
AKIDA_NUM_DEVICES=1
```

## Next Steps

1. **Add Your Data**: Place EEG data in `data/raw/`
2. **Train Models**: Use provided training scripts
3. **Test API**: Try the endpoints via Swagger UI
4. **Deploy**: Use Docker for deployment
5. **Monitor**: Check logs in `logs/` directory

## Getting Help

- Check `PROJECT_STRUCTURE.md` for detailed architecture
- Read `README.md` for project overview
- Explore API docs at http://localhost:8000/docs
- Review test files in `app/tests/` for usage examples

## Additional Resources

- **BrainChip Akida**: https://www.brainchip.com/
- **TVM Documentation**: https://tvm.apache.org/docs/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **PyTorch Documentation**: https://pytorch.org/docs/

---

For issues or questions, please refer to the project documentation or create an issue in the repository.

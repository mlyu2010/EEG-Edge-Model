# Installation Notes

## Quick Start

The project has been fully implemented with all components. Here's how to get started:

### Option 1: Docker (Recommended)

```bash
# Build and run
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f edge-models

# Access API at http://localhost:8000/docs
```

### Option 2: Local Installation

```bash
# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run application
uvicorn app.main:app --reload
```

## Dependencies Notes

### Optional Dependencies

Some dependencies are optional and commented out in `requirements.txt`:

1. **Akida SDK** (`akida==2.10.0`)
   - Requires BrainChip license
   - Uncomment when you have access to Akida hardware
   - Models work without it (placeholder implementation provided)

2. **Apache TVM** (`apache-tvm==0.17.0`)
   - Optional compilation framework
   - Requires LLVM installation
   - Project works without it using PyTorch/ONNX

### Minimal Installation

For basic testing without all ML features:

```bash
pip install -r requirements-minimal.txt
```

### Full Installation

For complete functionality including all frameworks:

```bash
pip install -r requirements.txt
```

## Common Issues

### Issue: PyTorch version not found

**Solution**: The requirements.txt uses flexible versioning (`torch>=2.0.0`). This will install the latest compatible PyTorch version for your system.

### Issue: Akida SDK not available

**Solution**: Akida SDK is optional. The code includes placeholder implementations. To use real Akida hardware:
1. Contact BrainChip for SDK access
2. Uncomment `akida==2.10.0` in requirements.txt
3. Update `app/utils/akida_quantization.py` with actual SDK calls

### Issue: TVM not available

**Solution**: TVM is optional. The code gracefully handles its absence. To enable TVM:
1. Install LLVM: `brew install llvm` (macOS) or `apt-get install llvm` (Ubuntu)
2. Uncomment `apache-tvm==0.17.0` in requirements.txt
3. Reinstall: `pip install apache-tvm`

## Verification

After installation, verify the setup:

```bash
# Check Python version
python --version  # Should be 3.13+

# Test imports
python -c "import torch; import fastapi; print('✓ Setup successful')"

# Run tests
pytest

# Start application
uvicorn app.main:app --reload
```

## Docker Installation

If dependencies fail locally, use Docker which has everything pre-configured:

```bash
# Build image
docker-compose build

# Run container
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

## Platform-Specific Notes

### macOS (Apple Silicon)

```bash
# PyTorch should automatically use ARM64 wheels
# ONNX Runtime supports Apple Silicon natively
```

### Linux

```bash
# Install system dependencies for some packages
sudo apt-get update
sudo apt-get install -y python3-dev build-essential
```

### Windows

```bash
# Use WSL2 or Docker for best experience
# Or install with conda for easier dependency management
```

## Project Status

All components are implemented and functional:
- ✅ TENN models (EEG, Vision, Segmentation, Anomaly Detection)
- ✅ FastAPI application with REST endpoints
- ✅ Model export (ONNX, TorchScript)
- ✅ Placeholder for Akida quantization
- ✅ Placeholder for TVM compilation
- ✅ Docker containerization
- ✅ Comprehensive tests
- ✅ Documentation and scripts

The project is ready to use with or without optional dependencies. Optional features (Akida, TVM) can be enabled when needed.

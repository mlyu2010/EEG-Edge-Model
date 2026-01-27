# Quick Reference Guide

## 🚀 Getting Started (30 seconds)

```bash
docker-compose build && docker-compose up -d
# Access: http://localhost:8000/docs
```

## 📦 Project Components

### Models (`app/models/`)
- `tenn_eeg.py` - EEG time series classification
- `vision_model.py` - Object detection/classification
- `segmentation_model.py` - Semantic segmentation
- `anomaly_detection.py` - Time series anomaly detection

### API Endpoints
- `GET /` - Root info
- `GET /health` - Health check
- `GET /api/v1/models` - List models
- `POST /api/v1/predict/eeg` - EEG inference
- `POST /api/v1/predict/vision` - Vision inference
- `POST /api/v1/predict/anomaly` - Anomaly detection
- `POST /api/v1/model/{name}/quantize` - Quantize model
- `GET /api/v1/benchmark/{name}` - Benchmark model

## 🛠️ Common Commands

### Docker
```bash
make docker-up        # Start containers
make docker-down      # Stop containers
make docker-logs      # View logs
make docker-restart   # Restart
```

### Development
```bash
make install          # Install dependencies
make run              # Run locally
make test             # Run all tests
make test-unit        # Unit tests only
make docs             # Generate docs
```

### Training & Export
```bash
make train            # Train EEG model
make export           # Export models
make benchmark        # Benchmark performance
```

## 📝 Files to Know

- `requirements.txt` - Python dependencies
- `requirements-minimal.txt` - Core dependencies only
- `.env.example` - Environment template
- `Makefile` - Command shortcuts
- `pytest.ini` - Test configuration
- `configs/model_config.yaml` - Model hyperparameters

## 🎯 Key Scripts

```bash
# Training (with device selection)
python scripts/train_eeg_model.py --epochs 100 --batch-size 32 --device auto  # Auto-detect
python scripts/train_eeg_model.py --epochs 100 --batch-size 32 --device mps   # Apple Silicon
python scripts/train_eeg_model.py --epochs 100 --batch-size 32 --device cuda  # NVIDIA GPU
python scripts/train_eeg_model.py --epochs 100 --batch-size 32 --device cpu   # CPU only

# Export
python scripts/export_model.py --model-type tenn_eeg --format onnx

# Benchmark (with device selection)
python scripts/benchmark.py --model-type tenn_eeg --iterations 1000 --device auto
python scripts/benchmark.py --model-type tenn_eeg --iterations 1000 --device mps
python scripts/benchmark.py --model-type tenn_eeg --iterations 1000 --device cuda

# Documentation
python scripts/generate_docs.py
```

## 🧪 Testing

```bash
pytest                    # All tests
pytest -m unit           # Unit tests
pytest -m integration    # Integration tests
pytest --cov=app         # With coverage
```

## 📚 Documentation

- `README.md` - Project overview
- `SETUP_GUIDE.md` - Detailed setup
- `PROJECT_STRUCTURE.md` - Architecture
- `IMPLEMENTATION_SUMMARY.md` - Features checklist
- `INSTALL_NOTES.md` - Troubleshooting

## 🔧 Configuration

### Environment Variables (`.env`)
```bash
APP_NAME=EEG-Edge-Model
DEBUG=True
PORT=8000
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=0.001
```

### Model Registry
Models available for export/benchmark:
- `tenn_eeg` - EEG classification
- `vision_classifier` - Image classification
- `segmentation` - Semantic segmentation
- `anomaly_detection` - Anomaly detection

## 🎨 API Examples

### List Models
```bash
curl http://localhost:8000/api/v1/models
```

### EEG Prediction
```bash
curl -X POST http://localhost:8000/api/v1/predict/eeg \
  -H "Content-Type: application/json" \
  -d '{"data": [[0.1, ...]], "model_type": "eeg"}'
```

### Benchmark
```bash
curl http://localhost:8000/api/v1/benchmark/tenn_eeg_v1?iterations=100
```

## 📊 Model Architectures

### TENN EEG
- Input: `(batch, 64, 256)` - 64 channels, 256 timesteps
- Output: `(batch, 4)` - 4 classes
- Components: Temporal Conv + State-Space + Classifier

### Vision Classifier
- Input: `(batch, 3, 224, 224)` - RGB images
- Output: `(batch, 1000)` - 1000 classes
- Architecture: Event-based Conv + ResNet blocks

### Segmentation
- Input: `(batch, 3, 256, 256)` - RGB images
- Output: `(batch, 21, 256, 256)` - 21 class masks
- Architecture: U-Net encoder-decoder

### Anomaly Detection
- Input: `(batch, 1, 256)` - Time series
- Output: Anomaly scores
- Architecture: Temporal autoencoder

## ⚡ Performance Tips

1. **Use Docker** for consistent environment
2. **Auto Device Detection**: Use `--device auto` to automatically select best hardware
3. **Apple Silicon**: Use `--device mps` on M1/M2/M3 Macs for ~2-5x speedup
4. **NVIDIA GPU**: Use `--device cuda` for ~10-100x speedup (requires CUDA toolkit)
5. **Batch Size**: Adjust based on available memory (larger on GPU)
6. **TVM Compilation**: Uncomment in requirements.txt for optimization
7. **Akida SDK**: Contact BrainChip for hardware acceleration

### Device Selection Quick Guide
- **Auto** (`auto`): Automatically selects MPS → CUDA → CPU
- **Apple Silicon** (`mps`): M1/M2/M3 Macs only, requires macOS 12.3+
- **NVIDIA GPU** (`cuda`): Requires CUDA toolkit installation
- **CPU** (`cpu`): Universal fallback, slowest option

## 🐛 Troubleshooting

### Port 8000 in use
```bash
# Change port in docker-compose.yml
ports: ["8080:8000"]
```

### Import errors
```bash
pip install -r requirements.txt
# Or minimal: pip install -r requirements-minimal.txt
```

### Docker build fails
```bash
docker-compose down -v
docker-compose build --no-cache
```

### Check available devices
```bash
python -c "from app.utils.device import print_device_info; print_device_info()"
```

### MPS not working on Apple Silicon
```bash
# Requires macOS 12.3+ and PyTorch 2.0+
pip install --upgrade torch torchvision
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

## 🎓 Learning Path

1. Start with Docker: `docker-compose up -d`
2. Explore API: http://localhost:8000/docs
3. Read `SETUP_GUIDE.md` for details
4. Run tests: `pytest`
5. Train model: `python scripts/train_eeg_model.py`
6. Export: `python scripts/export_model.py`
7. Benchmark: `python scripts/benchmark.py`

## 📦 Optional Dependencies

Commented out in `requirements.txt`:
- `akida` - Requires BrainChip license
- `apache-tvm` - Requires LLVM installation

Project works without these. Enable when needed.

## 🔗 Quick Links

- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health: http://localhost:8000/health

---

**Need help?** Check the full documentation in `SETUP_GUIDE.md` or `PROJECT_STRUCTURE.md`

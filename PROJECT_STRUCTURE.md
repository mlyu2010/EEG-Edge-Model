# EEG Edge Model - Project Structure

## Overview

This project implements TENN (Temporal Event-based Neural Networks) and other AI models optimized for 
BrainChip's Akida hardware platform, focusing on edge deployment scenarios including EEG signal 
processing, computer vision, and anomaly detection.

## Directory Structure

```
EEG-Edge-Model/
├── app/                          # Main application code
│   ├── api/                      # FastAPI endpoints
│   │   ├── __init__.py
│   │   └── routes.py            # API route definitions
│   ├── core/                     # Core configuration
│   │   ├── __init__.py
│   │   └── config.py            # Pydantic settings
│   ├── models/                   # Neural network models
│   │   ├── __init__.py
│   │   ├── tenn_eeg.py          # TENN model for EEG
│   │   ├── vision_model.py      # Object detection/classification
│   │   ├── segmentation_model.py # Semantic segmentation
│   │   └── anomaly_detection.py # Time series anomaly detection
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── model_export.py      # ONNX/TorchScript export
│   │   ├── akida_quantization.py # Akida quantization
│   │   ├── tvm_compiler.py      # TVM compilation
│   │   └── training.py          # Training utilities
│   ├── tests/                    # Test suite
│   │   ├── unit/                # Unit tests
│   │   │   ├── test_models.py
│   │   │   └── test_utils.py
│   │   └── integration/         # Integration tests
│   │       └── test_api.py
│   └── main.py                  # FastAPI application entry point
├── configs/                      # Configuration files
│   └── model_config.yaml        # Model configurations
├── data/                         # Data directories
│   ├── raw/                     # Raw data
│   └── processed/               # Processed data
├── models/                       # Model storage
│   ├── trained/                 # Trained models
│   ├── quantized/               # Quantized models
│   └── exported/                # Exported models
├── scripts/                      # Utility scripts
│   ├── generate_docs.py         # Documentation generation
│   ├── train_eeg_model.py       # Training script
│   ├── export_model.py          # Model export script
│   └── benchmark.py             # Benchmarking script
├── logs/                         # Application logs
├── Dockerfile                    # Docker image definition
├── docker-compose.yml           # Development environment
├── docker-compose.prod.yml      # Production environment
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
├── .env.example                 # Environment variables template
└── README.md                    # Project documentation
```

## Key Components

### 1. Models (`app/models/`)

#### TENN EEG Model
- **File**: `tenn_eeg.py`
- **Purpose**: Time series classification for EEG signals
- **Features**:
  - Temporal convolutional blocks
  - State-space recurrent layers
  - Optimized for edge deployment

#### Vision Model
- **File**: `vision_model.py`
- **Purpose**: Object detection and classification
- **Features**:
  - Event-based 2D convolutions
  - ResNet-style architecture
  - Detection heads for bounding boxes

#### Segmentation Model
- **File**: `segmentation_model.py`
- **Purpose**: Semantic segmentation
- **Architecture**: U-Net style encoder-decoder

#### Anomaly Detection
- **File**: `anomaly_detection.py`
- **Purpose**: Time series anomaly detection
- **Architecture**: Autoencoder-based

### 2. Utilities (`app/utils/`)

#### Model Export
- ONNX export with verification
- TorchScript compilation
- Model quantization

#### Akida Quantization
- Placeholder for BrainChip Akida SDK integration
- Quantization workflow
- Performance benchmarking

#### TVM Compiler
- Cross-platform compilation
- Auto-tuning support
- Multi-target optimization

#### Training
- Generic trainer class
- Metric computation
- Checkpointing and early stopping

### 3. API (`app/api/`)

RESTful API endpoints for:
- Model inference
- Model management
- Performance benchmarking
- Quantization services

### 4. Scripts (`scripts/`)

#### Training
```bash
python scripts/train_eeg_model.py --epochs 100 --batch-size 32
```

#### Export
```bash
python scripts/export_model.py --model-type tenn_eeg --format onnx
```

#### Benchmark
```bash
python scripts/benchmark.py --model-type tenn_eeg --iterations 1000
```

#### Documentation
```bash
python scripts/generate_docs.py
```

## Docker Deployment

### Development
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
docker-compose logs -f edge-models
```

### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## API Access

Once running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Testing

Run all tests:
```bash
pytest
```

Run specific test categories:
```bash
pytest -m unit
pytest -m integration
```

Run with coverage:
```bash
pytest --cov=app --cov-report=html
```

## Model Training Workflow

1. **Prepare Data**: Place data in `data/raw/`
2. **Train Model**: Use `scripts/train_eeg_model.py`
3. **Export Model**: Use `scripts/export_model.py`
4. **Quantize**: Use API endpoint or Akida SDK
5. **Deploy**: Use Docker or direct deployment

## TVM Compilation Workflow

1. Train PyTorch model
2. Compile with TVM for target hardware
3. Auto-tune for optimal performance
4. Benchmark across platforms
5. Deploy optimized binary

## Akida Deployment Notes

The project includes placeholder code for Akida integration. To use actual Akida hardware:

1. Install BrainChip Akida SDK
2. Update `app/utils/akida_quantization.py` with SDK calls
3. Run quantization workflow
4. Deploy to Akida device

## Performance Metrics

The framework tracks:
- Inference latency
- Throughput (samples/sec)
- Memory usage
- Cross-platform consistency
- Model compression ratio

## Development Tips

1. **Virtual Environment**: Always use Python 3.13+ virtual environment
2. **Dependencies**: Install from `requirements.txt`
3. **Configuration**: Copy `.env.example` to `.env` and customize
4. **Testing**: Write tests for new features
5. **Documentation**: Update docstrings for API documentation

## Next Steps

1. Add real EEG dataset loaders
2. Implement actual Akida SDK integration
3. Add more model architectures
4. Enhance API with authentication
5. Add monitoring and logging
6. Create deployment guides for specific hardware

## Contributing

When adding new models:
1. Create model class in `app/models/`
2. Add training script in `scripts/`
3. Update `MODEL_REGISTRY` in export/benchmark scripts
4. Add tests in `app/tests/`
5. Update documentation

## License

MIT

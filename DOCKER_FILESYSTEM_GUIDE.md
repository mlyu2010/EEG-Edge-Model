# Docker Filesystem Access Guide

## Quick Commands

### 1. View Files from Host Machine

Since docker-compose mounts volumes, files are synced between host and container:

```bash
# View from your host machine (same as container)
ls -la ./models/exported/
ls -la ./models/quantized/
ls -la ./models/trained/
ls -la ./data/

# Find all ONNX models
find ./models -name "*.onnx"

# Find all PyTorch models
find ./models -name "*.pt"
```

### 2. View Files Inside Container

```bash
# List models directory
docker exec edge-models-dev ls -la /app/models/

# List exported models with sizes
docker exec edge-models-dev ls -lh /app/models/exported/

# List quantized models
docker exec edge-models-dev ls -lh /app/models/quantized/

# Find all model files
docker exec edge-models-dev find /app/models -type f

# View file contents (for text files)
docker exec edge-models-dev cat /app/configs/model_config.yaml
```

### 3. Interactive Shell

```bash
# Open bash inside container
docker exec -it edge-models-dev bash

# Now you're inside the container, explore freely:
cd /app/models
ls -la
tree models/  # if tree is installed
find . -name "*.onnx"
du -sh *      # Check sizes

# Exit when done
exit
```

### 4. Copy Files from Container to Host

```bash
# Copy a single file
docker cp edge-models-dev:/app/models/exported/tenn_eeg.onnx ./local_copy.onnx

# Copy entire directory
docker cp edge-models-dev:/app/models/exported/ ./exported_models/

# Copy from host to container
docker cp ./mymodel.onnx edge-models-dev:/app/models/exported/
```

### 5. View Logs

```bash
# View recent logs
docker-compose logs --tail=50 edge-models

# Follow logs in real-time
docker-compose logs -f edge-models

# View all logs
docker logs edge-models-dev
```

## Directory Structure Inside Container

```
/app/
├── models/
│   ├── trained/          # Trained PyTorch models
│   │   ├── best_model.pt
│   │   ├── tenn_eeg_final.pt
│   │   └── checkpoint_epoch_*.pt
│   ├── exported/         # Exported ONNX/TorchScript models
│   │   ├── tenn_eeg.onnx
│   │   ├── tenn_eeg_akida.onnx
│   │   └── tenn_eeg_torchscript.pt
│   └── quantized/        # Quantized models for edge deployment
│       └── (quantized models will appear here)
├── data/
│   ├── raw/             # Raw input data
│   └── processed/       # Preprocessed data
├── logs/                # Application logs
├── configs/             # Configuration files
└── app/                 # Application source code
```

## Current Models Available

Based on the container inspection:

### Exported Models (/app/models/exported/)
```
tenn_eeg.onnx                  - 2.6 MB (ONNX format)
tenn_eeg.onnx.data            - 25 MB  (ONNX weights)
tenn_eeg_akida.onnx           - 2.6 MB (Akida format)
tenn_eeg_akida.onnx.data      - 25 MB
tenn_eeg_torchscript.pt       - 719 KB (TorchScript)
```

### Trained Models (/app/models/trained/)
```
tenn_eeg_final.pt             - Final trained model
best_model.pt                 - Best checkpoint
checkpoint_epoch_5.pt         - Training checkpoint
checkpoint_epoch_10.pt        - Training checkpoint
```

## Inspecting Model Files

### Using Python Inside Container

```bash
docker exec -it edge-models-dev python3 << 'EOF'
import torch
import onnx

# Load PyTorch model info
model_path = "/app/models/trained/tenn_eeg_final.pt"
state_dict = torch.load(model_path, map_location='cpu')
print(f"PyTorch model keys: {list(state_dict.keys())[:5]}")

# Load ONNX model info
onnx_path = "/app/models/exported/tenn_eeg.onnx"
onnx_model = onnx.load(onnx_path)
print(f"ONNX model version: {onnx_model.ir_version}")
print(f"ONNX inputs: {[i.name for i in onnx_model.graph.input]}")
print(f"ONNX outputs: {[o.name for o in onnx_model.graph.output]}")
EOF
```

### View Model Metadata

```bash
# ONNX model inspection
docker exec edge-models-dev python3 -c "
import onnx
model = onnx.load('/app/models/exported/tenn_eeg.onnx')
print('Input shape:', [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim])
print('Output shape:', [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim])
"
```

## Disk Usage

```bash
# Check container disk usage
docker exec edge-models-dev du -sh /app/models/*

# Check specific directory sizes
docker exec edge-models-dev du -h /app/models/exported/
docker exec edge-models-dev df -h
```

## Working with Volumes

### Current Volume Mounts (from docker-compose.yml)

```yaml
volumes:
  - ./app:/app/app              # Source code (hot reload)
  - ./data:/app/data            # Data directory
  - ./models:/app/models        # Models directory
  - ./logs:/app/logs            # Logs directory
  - ./configs:/app/configs      # Configuration
```

This means:
- ✅ Files are **bidirectionally synced**
- ✅ Changes in container → visible on host
- ✅ Changes on host → visible in container
- ✅ No need to rebuild container for data changes

### View Volume Contents from Host

```bash
# These are the SAME files as in the container
ls -lh models/exported/
ls -lh models/quantized/
ls -lh models/trained/
```

## Creating/Quantizing Models

### Using API Endpoints

```bash
# Quantize a model (currently placeholder)
curl -X POST "http://localhost:8000/api/v1/model/tenn_eeg.onnx/quantize?target=onnx"
```

### Using Scripts Inside Container

```bash
# Export a model
docker exec edge-models-dev python scripts/export_model.py \
  --model-type tenn_eeg \
  --format onnx

# Train a model
docker exec edge-models-dev python scripts/train_eeg_model.py \
  --epochs 10 \
  --batch-size 32

# Benchmark
docker exec edge-models-dev python scripts/benchmark.py \
  --model-type tenn_eeg \
  --iterations 100
```

## Troubleshooting

### "Permission Denied" Issues

```bash
# Fix permissions from host
chmod -R 755 models/
chown -R $USER:$USER models/

# Or from container (as root)
docker exec -u root edge-models-dev chown -R root:root /app/models/
```

### "No Space Left" Issues

```bash
# Check disk usage
docker system df

# Clean up unused images/containers
docker system prune -a

# Check container size
docker exec edge-models-dev df -h
```

### View What Changed

```bash
# See filesystem changes in container
docker diff edge-models-dev
```

## Pro Tips

### 1. Create a Shell Alias

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias docker-shell='docker exec -it edge-models-dev bash'
alias docker-ls='docker exec edge-models-dev ls -lah'
alias docker-models='docker exec edge-models-dev ls -lh /app/models/exported/'
```

Then use:
```bash
docker-shell      # Jump into container
docker-models     # Quick model listing
```

### 2. Watch for Changes

```bash
# Monitor directory for changes
watch -n 2 'docker exec edge-models-dev ls -lh /app/models/quantized/'

# Or on host
watch -n 2 'ls -lh models/quantized/'
```

### 3. Real-time Log Monitoring

```bash
# Follow logs with grep filter
docker-compose logs -f edge-models | grep "quantiz"
```

### 4. Execute Python in Container

```bash
# Quick Python commands
docker exec edge-models-dev python -c "import torch; print(torch.__version__)"

# Run a script
docker exec edge-models-dev python /app/scripts/export_model.py --help
```

## Summary

**Easiest ways to see files:**

1. **From host**: `ls -lh ./models/exported/` (files are synced)
2. **Quick peek**: `docker exec edge-models-dev ls -lh /app/models/`
3. **Explore**: `docker exec -it edge-models-dev bash` then navigate
4. **Copy out**: `docker cp edge-models-dev:/app/models/exported/ ./`

All files in mounted volumes are accessible from both host and container! 🎉

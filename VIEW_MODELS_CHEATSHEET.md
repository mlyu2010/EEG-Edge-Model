# Quick Cheatsheet: View Models in Docker

## ⚡ Fastest Way (From Your Computer)

Since files are volume-mounted, you can access them directly:

```bash
# View all exported models
ls -lh models/exported/

# View quantized models
ls -lh models/quantized/

# View trained models
ls -lh models/trained/

# Open in Finder (macOS)
open models/exported/

# Open in file browser (Linux)
xdg-open models/exported/
```

## 🐳 View Inside Docker Container

```bash
# Quick listing
docker exec edge-models-dev ls -lh /app/models/exported/

# Interactive exploration
docker exec -it edge-models-dev bash
# Then: cd /app/models && ls -la
```

## 📊 Current Models (as of now)

### ✅ Exported Models (`models/exported/`)
```
tenn_eeg.onnx              2.5 MB    ONNX format
tenn_eeg.onnx.data         24 MB     ONNX weights
tenn_eeg_akida.onnx        2.6 MB    Akida ONNX
tenn_eeg_akida.onnx.data   24 MB     Akida weights
tenn_eeg_torchscript.pt    718 KB    TorchScript
```

### 📁 Directory Locations

| Location | Host Path | Container Path |
|----------|-----------|----------------|
| Exported | `./models/exported/` | `/app/models/exported/` |
| Quantized | `./models/quantized/` | `/app/models/quantized/` |
| Trained | `./models/trained/` | `/app/models/trained/` |
| Data | `./data/` | `/app/data/` |
| Logs | `./logs/` | `/app/logs/` |

## 🔍 Inspect Model Details

### ONNX Model
```bash
# From host (if onnx installed)
python3 -c "import onnx; m=onnx.load('models/exported/tenn_eeg.onnx'); print(m.graph.input[0])"

# From container
docker exec edge-models-dev python3 -c "import onnx; m=onnx.load('/app/models/exported/tenn_eeg.onnx'); print('Inputs:', [i.name for i in m.graph.input]); print('Outputs:', [o.name for o in m.graph.output])"
```

### PyTorch Model
```bash
# From container
docker exec edge-models-dev python3 << 'EOF'
import torch
model = torch.load('/app/models/trained/best_model.pt', map_location='cpu')
print(f"Checkpoint contains: {list(model.keys())}")
print(f"Epoch: {model.get('epoch', 'N/A')}")
print(f"Loss: {model.get('loss', 'N/A')}")
EOF
```

## 📤 Copy Models Out of Docker

```bash
# Copy single file
docker cp edge-models-dev:/app/models/exported/tenn_eeg.onnx ./my_model.onnx

# Copy entire directory
docker cp edge-models-dev:/app/models/exported/ ./exported_backup/
```

## 🎯 Key Takeaway

**Files are synced!** Anything you see in `models/` on your computer is the same as `/app/models/` in the container.

```bash
# These show the SAME files:
ls -lh models/exported/                            # Host
docker exec edge-models-dev ls -lh /app/models/exported/  # Container
```

## 🛠️ Create New Models

### Via API
```bash
curl -X POST "http://localhost:8000/api/v1/model/tenn_eeg_v1/quantize?target=onnx"
```

### Via Scripts
```bash
# Export model
docker exec edge-models-dev python scripts/export_model.py \
  --model-type tenn_eeg --format onnx

# Then check:
ls -lh models/exported/
```

## 💡 Pro Tips

1. **Use your regular file browser**: Files are on your computer at `models/`
2. **Volume mounts = Real-time sync**: Changes anywhere appear everywhere
3. **No rebuild needed**: Add files to `models/` and they're instantly in the container
4. **Safe to delete**: Just delete files from `models/` if you don't need them

## 🚀 Quick Commands

```bash
# See everything
find models/ -type f

# Count models
find models/ -name "*.onnx" -o -name "*.pt" | wc -l

# Check sizes
du -sh models/*

# Watch for new files
watch -n 2 'ls -lh models/quantized/'
```

---

**TL;DR**: Just use `ls -lh models/exported/` on your computer! 🎉

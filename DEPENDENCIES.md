# Dependencies

## Inference-Only (build & run C++ examples from existing .engine files)

| Dependency | Purpose |
|---|---|
| g++ (C++17) | Compiler |
| CUDA Toolkit 11.8+ (`nvcc`, `cudart`, `cuda`) | GPU runtime and kernel compilation |
| TensorRT 8 (`libnvinfer`) | Engine loading and inference |
| OpenCV 4 (`opencv_core`, `opencv_imgproc`, `opencv_imgcodecs`) | Image I/O and preprocessing |
| libdl, libstdc++ | System libraries |

### Runtime Assets

- `workspace/models/*.engine` — pre-built TensorRT engines (yolo11n.engine, dfine_n_coco.engine)
- `workspace/inference/*.jpg` — test input images
- `workspace/result/` — output directory for annotated results

## ONNX-to-TensorRT Export-Only

These are only needed to regenerate `.engine` files from ONNX models.

| Dependency / Asset | Purpose |
|---|---|
| `trtexec` (TensorRT CLI tool) | Converts ONNX models to TensorRT engines |
| Python 3 + `pip` | Required for running transformation scripts |
| `onnx` Python package (`pip install onnx`) | ONNX model manipulation |
| `workspace/v8trans.py` | YOLO11 ONNX pre-processing transformation script |
| ONNX source models (`yolo11n.onnx`, `dfine_n_coco.onnx`) | Input models for engine generation |

### Export Workflow

```bash
# 1. Transform YOLO11 ONNX (adds decode heads)
python3 workspace/v8trans.py yolo11n.onnx yolo11n.transd.onnx

# 2. Build TensorRT engines
trtexec --onnx=yolo11n.transd.onnx --saveEngine=workspace/models/yolo11n.engine --fp16
trtexec --onnx=dfine_n_coco.onnx --saveEngine=workspace/models/dfine_n_coco.engine --fp16
```

## Removed Dependencies (no longer required)

The following were removed from the build; they are **not** needed:

- FreeType / `libfreetype` — previously used for OSD text rendering (replaced by `src/common/draw.hpp`)
- `opencv_videoio` — no video capture in current examples
- `nvonnxparser` — not used at inference time; `trtexec` handles ONNX parsing during export
- `nvinfer_plugin` — not used in current code
- `cublas` — not used in current code
- OpenMP (`-fopenmp`) — not used in current code
- OSD / workspace font files — removed; drawing uses inline `draw.hpp` with OpenCV

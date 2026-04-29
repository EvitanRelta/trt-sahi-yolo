# Work Done: TRT-SAHI-YOLO for YOLO11 + D-FINE on TensorRT 8.5.2

## Objective

Get the `trt-sahi-yolo` repository working for **YOLO11 (detection only)** and **D-FINE** on an environment with:
- CUDA 11.8 at `/usr/local/cuda`
- TensorRT 8.5.2.2
- OpenCV 4.2.0
- NVIDIA T600 Laptop GPU (sm_75, ~4GB VRAM)
- C++ only — no Python bindings needed

Both standard inference and SAHI (Slice and Stitch) variants were requested, with non-SAHI prioritized first.

---

## 1. Dependency Installation

```bash
apt-fast update
apt-fast install -y python3-pip
pip3 install onnx
```

- `onnx` Python package — needed by `v8trans.py` to transpose YOLO11 ONNX output
- **FreeType, fonts, and OSD are no longer required** (see Section 9 for details). Drawing is now done with OpenCV's built-in Hershey fonts.

---

## 2. Source Code Cleanup

### Files Deleted

| Category | Files |
|----------|-------|
| Python bindings | `src/interface.cpp`, `src/pybind11/` (entire directory) |
| Unused YOLO variants | `src/trt/yolo/yolov5.hpp`, `yolov5.cu`, `yolo11seg.hpp`, `yolo11seg.cu`, `yolo11pose.hpp`, `yolo11pose.cu`, `yolo11obb.hpp`, `yolo11obb.cu` |
| Unused SAHI variants | `src/trt/sahiyolo/yolov5_sahi.hpp/cu`, `yolo11_seg_sahi.hpp/cu`, `yolo11_pose_sahi.hpp/cu`, `yolo11_obb_sahi.hpp/cu` |
| Unused examples | `src/examples/yolov5/`, `src/examples/yolo11seg/`, `src/examples/yolo11pose/`, `src/examples/yolo11obb/` |
| Python demo | `workspace/demo.py` |

### Files Kept

| Category | Files |
|----------|-------|
| YOLO11 detection | `src/trt/yolo/yolo.hpp`, `yolo11.hpp`, `yolo11.cu` |
| YOLO11 SAHI | `src/trt/sahiyolo/sahiyolo.hpp`, `yolo11_sahi.hpp`, `yolo11_sahi.cu` |
| D-FINE | `src/trt/dfine/dfine.hpp`, `dfine.cu` |
| D-FINE SAHI | `src/trt/dfinesahi/dfinesahi.hpp`, `dfinesahi.cu` |
| SAHI slicing | `src/trt/slice/` (all files) |
| Common infra | `src/common/` (all files, including `draw.hpp`) |
| CUDA kernels | `src/kernels/` (all files) |
| Examples | `src/examples/yolo11/`, `src/examples/dfine/` |

### Source Code Trimming

**`src/trt/infer.hpp`** — `ModelType` enum trimmed from 12 values to 4:
```cpp
enum class ModelType : int {
    YOLO11    = 2,
    YOLO11SAHI = 3,
    DFINE     = 10,
    DFINESAHI = 11
};
```

**`src/trt/infer.cpp`** — Includes and switch cases trimmed to match. Only includes `yolo11.hpp`, `dfine.hpp`, `dfinesahi.hpp`, `yolo11_sahi.hpp`.

**`src/main.cpp`** — Only calls `run_yolo11()`, `run_yolo11_sahi()`, `run_dfine()`, `run_dfine_sahi()`.

**Example files** — Updated engine paths and added error handling (null checks, empty result checks).

---

## 3. Makefile Rewrite

Key changes from the original Makefile (which targeted CUDA 12 + TRT 10):

| Setting | Old | New |
|---------|-----|-----|
| `cuda_home` | `/usr/local/cuda-12` | `/usr/local/cuda` |
| `cuda_arch` | `8.6` | `75` (sm_75 for T600) |
| `TRT_VERSION` | `10` | `8` |
| OpenCV include | Hardcoded path | `pkg-config --cflags opencv4` |
| OpenCV lib | Hardcoded path | `pkg-config --libs opencv4` |
| TRT include | `/opt/nvidia/TensorRT-10.x/include` | `/usr/include/x86_64-linux-gnu` |
| TRT lib | `/opt/nvidia/TensorRT-10.x/lib` | `/usr/lib/x86_64-linux-gnu` |
| Link TRT libs | `nvinfer nvinfer_plugin nvonnxparser` | `nvinfer nvonnxparser` (removed `nvinfer_plugin`) |
| CUDA link | `cuda cublas cudart cudnn` | `cuda cudart` (removed `cublas`, `cudnn`) |
| OpenCV lib | (includes `opencv_videoio`) | removed `opencv_videoio` |
| OpenMP | `-fopenmp` in compile/link flags | removed |
| Python include/lib | Present | Removed |
| Targets | `trtsahi.so` (shared lib) + `pro` | `pro` only |

### Build fixes applied during compilation

1. **nvcc C++ standard flag**: `-std=c++17` must be passed directly to nvcc, not just via `-Xcompiler`. Otherwise nvcc's frontend uses C++14 and can't parse C++17 STL headers.
   ```makefile
   # Before (broken):
   cu_compile_flags := -Xcompiler "$(cpp_compile_flags)"
   # After (fixed):
   cu_compile_flags := -std=$(stdcpp) -Xcompiler "-w -g -O0 -m64 -fPIC -fopenmp -pthread $(include_paths)"
   ```

2. **ONNX parser library name**: The installed library is `libnvonnxparser.so` (with `nv` prefix), not `libonnxparser.so`.

### Build command

```bash
make clean && make pro -j$(nproc)
```

Output: `workspace/pro` (~9MB binary)

---

## 4. TensorRT Engine Export

### YOLO11 Engine

#### Problem: output shape transposition

The original `yolo11n.trt` was built from the un-transposed ONNX, producing output shape `1x84x8400` instead of `1x8400x84`. This caused `num_classes_ = bbox_head_dims_[2] - 4 = 8400 - 4 = 8396` (instead of 80), leading to out-of-bounds memory access and crashes.

YOLO11/YOLOv8 ONNX models output in channels-first format `[batch, 4+num_classes, num_anchors]`. The decode kernel expects anchors-first `[batch, num_anchors, 4+num_classes]`.

#### Step 1: Transpose the ONNX

```bash
python3 workspace/v8trans.py yolo11n.onnx
```

This adds a `Transpose` node to the ONNX graph. The script outputs `yolo11n.transd.onnx` (ignores the second CLI argument; always writes to `<prefix>.transd.onnx`).

#### Step 2: Build engine with trtexec

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolo11n.transd.onnx \
  --saveEngine=workspace/models/yolo11n.engine \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640
```

**Result:**
- Output shape: `1x8400x84` (correct)
- Throughput: **218.20 qps**
- Mean latency: **5.32 ms**

### D-FINE Engine

#### Inspection

```bash
python3 -c "
import onnx
model = onnx.load('dfine_n_coco.onnx')
for inp in model.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f'Input: {inp.name}, shape: {shape}, dtype: {inp.type.tensor_type.elem_type}')
"
```

Output:
```
Input: images, shape: [0, 3, 640, 640], dtype: 1 (float32)
Input: orig_target_sizes, shape: [0, 2], dtype: 7 (int64)
```

#### Export command

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=dfine_n_coco.onnx \
  --saveEngine=workspace/models/dfine_n_coco.engine \
  --fp16 \
  --minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --optShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --maxShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --memPoolSize=workspace:512
```

**Result:**
- Throughput: **118.24 qps**
- Mean latency: **8.85 ms**

### Flag explanation

| Flag | Purpose |
|------|---------|
| `--fp16` | Enable FP16 inference. ~2x throughput with minimal accuracy loss. |
| `--minShapes` / `--optShapes` / `--maxShapes` | Define dynamic shape range. When min=opt=max, TRT builds a **static** engine (faster, less memory). |
| `--memPoolSize=workspace:512` | Limit TRT workspace to 512 MiB. Needed for D-FINE on T600 to avoid OOM during engine build. |
| Input names (`images`, `orig_target_sizes`) | Must match the ONNX model's input tensor names exactly. |
| `orig_target_sizes:1x2` | D-FINE's second input — original image width/height for box coordinate scaling. |

### Why batch=1 for both engines

The T600 has ~4GB VRAM. When building with dynamic batch ranges (e.g., max=16 or max=32), TRT:
1. Allocates input/output buffers sized for maxShapes
2. Profiles multiple kernel implementations during build

Batch=16 for D-FINE caused OOM during trtexec's profiling phase. Batch=1 succeeded with `--memPoolSize=workspace:512`.

---

## 5. Current Status

### Working ✅

| Feature | Status | Notes |
|---------|--------|-------|
| YOLO11 standard inference | ✅ Working | `run_yolo11()` — 218 qps, result saved to `result/yolo11.jpg` |
| D-FINE standard inference | ✅ Working | `run_dfine()` — 118 qps, result saved to `result/dfine.jpg` |
| Drawing/visualization | ✅ Working | Uses OpenCV `cv::rectangle` + `cv::putText` (Hershey font) via `src/common/draw.hpp` |
| Build system | ✅ Working | `make pro -j$(nproc)` |

### Not Working ❌

| Feature | Status | Root Cause |
|---------|--------|------------|
| YOLO11 SAHI | ❌ Assertion failure | SAHI slices image into N patches, tries to run all at once, but engine has static batch=1 |
| D-FINE SAHI | ❌ Assertion failure | Same issue — `assert(num_image <= max_batch_size_)` fails when `num_image=6` |

---

## 6. Future Work

### 6.1 Fix SAHI Inference (High Priority)

**The problem:** Both SAHI implementations (`yolo11_sahi.cu`, `dfinesahi.cu`) slice the input image into multiple patches (e.g., 2x3=6 patches for a 1920x1080 image with 640x640 slices and 0.3 overlap ratio). They then try to run all patches through the TRT engine in a single batch. With static batch=1 engines, this fails because `num_image > max_batch_size_`.

**Two approaches to fix:**

#### Approach A: Chunked processing in code (recommended for T600)

Modify the SAHI `forwards()` functions to process patches in chunks of `max_batch_size_`:

```cpp
// In yolo11_sahi.cu forwards():
for (int chunk_start = 0; chunk_start < num_image; chunk_start += max_batch_size_) {
    int chunk_size = min(max_batch_size_, num_image - chunk_start);
    // Set run dims for this chunk
    // Preprocess chunk_size patches
    // Run TRT forward
    // Decode + NMS for this chunk
}
// Apply global NMS across all chunks
```

This approach works with the existing static batch=1 engines and is memory-safe.

#### Approach B: Build dynamic batch engines

For YOLO11:
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolo11n.transd.onnx \
  --saveEngine=workspace/models/yolo11n_b32.engine \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:8x3x640x640 \
  --maxShapes=images:32x3x640x640
```

For D-FINE:
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=dfine_n_coco.onnx \
  --saveEngine=workspace/models/dfine_n_coco_b32.engine \
  --fp16 \
  --minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --optShapes=images:8x3x640x640,orig_target_sizes:8x2 \
  --maxShapes=images:32x3x640x640,orig_target_sizes:32x2 \
  --memPoolSize=workspace:512
```

**T600 VRAM constraints:**
- Each 640x640 FP16 image = `3 × 640 × 640 × 2 bytes = 2.45 MB`
- Batch=32 input alone = ~78 MB
- TRT intermediate tensors + engine overhead can easily exceed 4GB
- May need to reduce max batch to 8 or 16 on the T600
- `--memPoolSize=workspace:512` helps limit workspace allocation

**Recommendation:** Approach A is safer on the T600 and doesn't require rebuilding engines. Approach B gives better throughput on GPUs with more VRAM.

### 6.2 D-FINE int64/int32 Mismatch (Medium Priority)

The D-FINE ONNX model uses `int64` for the `orig_target_sizes` input and `labels` output. The TRT8 code path in `dfine.cu` and `dfinesahi.cu` uses `int32_t`:

```cpp
#if NV_TENSORRT_MAJOR >= 10
    tensor::Memory<int64_t> input_buffer_orig_target_size_;
    tensor::Memory<int64_t> output_labels_;
#else
    tensor::Memory<int32_t> input_buffer_orig_target_size_;
    tensor::Memory<int32_t> output_labels_;
#endif
```

With TRT8, the code allocates `int32_t` buffers but the engine may expect `int64_t`. This hasn't caused visible crashes yet (TRT may do implicit casting, or the values fit in 32 bits), but could silently produce wrong results for very large images (>2^31 pixels in one dimension, unlikely in practice).

**To verify:** Print the actual binding dtypes from the TRT8 engine at runtime using `engine->getBindingDataType(i)` and compare with what the code allocates.

### 6.3 Performance Optimization (Low Priority)

- **Warm-up runs:** The examples already include warm-up loops (5 iterations) before benchmarking
- **CUDA graphs:** TRT 8.5+ supports CUDA graphs for reduced launch overhead
- **Multi-stream inference:** Can overlap preprocessing/inference/postprocessing across different images
- **INT8 quantization:** Further throughput gains with calibration dataset

---

## 7. Environment Reference

```
OS:             Ubuntu 20.04 (Docker container)
CUDA:           11.8.0 at /usr/local/cuda
TensorRT:       8.5.2.2
cuDNN:          8.6.0.163
OpenCV:         4.2.0
GPU:            NVIDIA T600 Laptop (sm_75, ~4GB VRAM)
Compiler:       g++ (C++17), nvcc
trtexec:        /usr/src/tensorrt/bin/trtexec
```

## 8. Key File Paths

```
/workspace/trt-sahi-yolo/
├── Makefile                              # Build system (TRT8 mode)
├── DEPENDENCIES.md                       # Dependency inventory (inference vs export-only)
├── src/
│   ├── main.cpp                          # Entry point
│   ├── common/draw.hpp                   # Header-only OpenCV drawing (boxes + labels)
│   ├── trt/infer.hpp                     # ModelType enum + load() declaration
│   ├── trt/infer.cpp                     # Factory: routes ModelType to implementations
│   ├── trt/yolo/yolo.hpp                 # YOLO base class (TRT8/10 abstraction)
│   ├── trt/yolo/yolo11.hpp/cu            # YOLO11 detection
│   ├── trt/dfine/dfine.hpp/cu            # D-FINE detection
│   ├── trt/sahiyolo/sahiyolo.hpp         # SAHI YOLO base class
│   ├── trt/sahiyolo/yolo11_sahi.hpp/cu   # YOLO11 SAHI (broken: batch limit)
│   ├── trt/dfinesahi/dfinesahi.hpp/cu    # D-FINE SAHI (broken: batch limit)
│   └── trt/slice/                        # SAHI image slicing logic
├── workspace/
│   ├── pro                               # Compiled C++ binary
│   ├── models/
│   │   ├── yolo11n.engine                # TRT8 engine (1x8400x84, static batch=1)
│   │   └── dfine_n_coco.engine           # TRT8 engine (static batch=1)
│   ├── inference/                        # Test images
│   ├── result/                           # Output images
│   └── v8trans.py                        # ONNX transpose script for YOLO11
├── yolo11n.onnx                          # Original YOLO11 ONNX
├── yolo11n.transd.onnx                   # Transposed YOLO11 ONNX (v8trans.py output)
├── dfine_n_coco.onnx                     # D-FINE ONNX
├── coco.names                            # 80 COCO class names
└── WORK_DONE.md                          # This file
```

---

## 9. Dependency Cleanup (2026-04-29)

Removed FreeType, external fonts, and OSD overlay code. Replaced with minimal OpenCV-only drawing.

### OSD/FreeType/font removal

- **Deleted** `src/osd/` entirely: `osd.hpp`, `osd.cpp`, `cvx_text.hpp`, `cvx_text.cpp`, `labelLayout.hpp`, `position.hpp`.
- **Deleted** `workspace/font/SIMKAI.TTF` and the empty `workspace/font/` directory.
- **Added** `src/common/draw.hpp` — header-only `draw::draw_detections(cv::Mat&, const DetectionBoxArray&)` using `cv::rectangle` and `cv::putText` (Hershey font). No external font files needed.
- **Updated** `src/examples/yolo11/yolo11.cpp` and `src/examples/dfine/dfine.cpp` to `#include "common/draw.hpp"` and call `draw::draw_detections` instead of `osd()`.

### Makefile link-flag simplification

| Removed | Reason |
|---------|--------|
| FreeType include/link (`/usr/include/freetype2`, `-lfreetype`) | OSD deleted |
| `opencv_videoio` | Not used by remaining code |
| `nvonnxparser` removed from link, then restored | Only `nvinfer` + `nvonnxparser` kept; `nvinfer_plugin` dropped |
| `cublas` from CUDA link flags | Not required by inference path |
| `-fopenmp` from compile and link flags | Not used |

Build verified: `make clean && make pro -j$(nproc)` succeeds.

### Dead CUDA kernel/wrapper code removed

| Symbol / wrapper | File(s) |
|------------------|---------|
| `decode_kernel_v5` / `decode_kernel_invoker_v5` | `src/kernels/decode.cu`, `decode.cuh` |
| YOLO11 pose decode/NMS wrappers | `src/kernels/kernel_warp.cu`, `kernel_warp.hpp` |
| YOLO11 OBB decode/NMS wrappers + OBB-only helpers | `src/kernels/kernel_warp.cu`, `kernel_warp.hpp` |
| `decode_single_mask_kernel` / `decode_single_mask_invoker` | `src/kernels/decode.cu`, `decode.cuh` |

### Stale workspace files removed

- `workspace/trtsahi.pyi` (Python stub, no longer needed)
- `yolo11n.trt` (old engine file in repo root)

### New file added

- `DEPENDENCIES.md` — separates inference-only from ONNX→TRT export-only dependencies; documents that FreeType/fonts/OSD are no longer required.

### Verification

`make clean && make pro -j$(nproc)` succeeded. Running `./pro` from `workspace/` produced result images with 20 filtered detections each for YOLO11 and D-FINE. No SAHI static-batch issue in this run.

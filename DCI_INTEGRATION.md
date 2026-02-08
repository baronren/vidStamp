# DCI-Compliant Forensic Watermarking System

## Overview

This repository provides a **DCI CTP-compliant forensic marking (FM) system** based on the StegaStamp watermarking architecture. The system is designed to run within Digital Cinema Media Blocks (MB) in FIPS 140-2 Level 3 secure environments, supporting real-time watermark embedding at 2K/4K resolutions and 24fps/48fps frame rates.

## Architecture

### System Components

1. **Payload Structure (`dci_payload.py`)**
   - 48-bit DCI-compliant payload:
     - 16-bit timestamp (increments every 15 minutes, yearly reset)
     - 20-bit Forensic Marking ID (FMID) - supports 1,048,576 unique locations
     - 12-bit CRC error correction code
   - Designed for extraction from any 5-minute video segment

2. **C++ Interface (`forensic_watermarker.h`)**
   - `ForensicWatermarker` class for Media Block integration
   - Zero-copy frame processing
   - Thread-safe single producer/consumer design
   - Support for DCI 2K (2048x1080) and 4K (4096x2160)

3. **Ring Buffer (`ring_buffer.h`)**
   - Circular buffer for temporal consistency
   - Maintains sliding window of frames for 3D convolution
   - GPU-friendly memory layout
   - Zero-copy operations

4. **Model Optimizer (`model_optimizer.py`)**
   - Converts TensorFlow model to ONNX/TensorRT
   - Removes diffusion dependencies
   - Applies FP16/INT8 quantization
   - Extracts encoder-only model for real-time inference

## Key Requirements

### 1. Payload Design

The system embeds a complete 48-bit payload in **every frame**, ensuring:
- **5-minute extractability**: Any 5-minute segment (7,200 frames @ 24fps) contains thousands of complete payloads
- **Robustness**: Resistant to frame drops, compression, and temporal attacks
- **Error correction**: 12-bit CRC enables detection and correction of bit errors

### 2. Real-Time Performance

The original StegaStamp model is too slow for real-time use. Our optimization strategy:

```
Original: Diffusion-based generation (several seconds per frame)
         ↓
Optimized: Single forward pass encoder (< 50ms per frame)
```

**Critical Transformation:**
- **Remove**: Diffusion process, iterative denoising
- **Keep**: Encoder-decoder architecture, residual addition
- **Add**: Temporal consistency via ring buffer
- **Export**: TensorRT with FP16 quantization

### 3. DCI Timestamp Logic

The timestamp follows DCI specifications:
- **Increment**: Every 15 minutes (96 intervals per day)
- **Reset**: Annually (at year boundary)
- **Range**: 0-65535 (16 bits, supporting ~683 days)

Example calculation:
```python
from dci_payload import DCIPayload
import datetime

# Current time
dt = datetime.datetime(2024, 6, 15, 14, 30)  # June 15, 2:30 PM
timestamp = DCIPayload.calculate_timestamp(dt)
# Result: Day 166, 14:30 → (165 * 96) + (14*60 + 30)//15 → 15898

# Decode back
day, hour, minute = DCIPayload.timestamp_to_datetime(15898)
# Result: (166, 14, 30)
```

### 4. Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Media Block (Secure)                    │
│                                                           │
│  ┌──────────────┐  Raw    ┌──────────────────────────┐ │
│  │  Decryptor   │─Frames─→│  Forensic Watermarker    │ │
│  │              │          │                          │ │
│  └──────────────┘          │  ┌────────────────────┐ │ │
│                            │  │   Ring Buffer       │ │ │
│                            │  │   (8 frames)        │ │ │
│                            │  └────────────────────┘ │ │
│                            │  ┌────────────────────┐ │ │
│                            │  │  TensorRT Engine    │ │ │
│                            │  │  (FP16/INT8)        │ │ │
│                            │  └────────────────────┘ │ │
│                            │  ┌────────────────────┐ │ │
│                            │  │  Payload Encoder    │ │ │
│                            │  │  (48-bit)           │ │ │
│                            │  └────────────────────┘ │ │
│                            └──────────────────────────┘ │
│                                     │                    │
│                                 Marked Frames            │
│                                     ↓                    │
│                            ┌──────────────────┐          │
│                            │    Projector     │          │
│                            └──────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

## Usage

### Python API

#### 1. Payload Encoding/Decoding

```python
from dci_payload import DCIPayload
import numpy as np

# Initialize with FMID (screen/location ID)
fmid = 12345  # Screen #12345
payload_encoder = DCIPayload(fmid)

# Encode payload with current timestamp
bits = payload_encoder.encode_payload()  # Returns 48-bit array

# Or encode for VidStamp (100-bit format)
vidstamp_bits = payload_encoder.encode_for_vidstamp()

# Decode payload
timestamp, fmid, crc, is_valid = DCIPayload.decode_payload(bits)

if is_valid:
    print(f"FMID: {fmid}, Timestamp: {timestamp}")
    day, hour, minute = DCIPayload.timestamp_to_datetime(timestamp)
    print(f"Time: Day {day} at {hour:02d}:{minute:02d}")
else:
    print("CRC check failed!")
```

#### 2. Model Optimization

```python
from model_optimizer import create_optimized_pipeline

# Convert TensorFlow model to optimized formats
success = create_optimized_pipeline(
    model_path='saved_models/stegastamp_pretrained',
    output_dir='optimized_models',
    precision='fp16'  # or 'fp32', 'int8'
)
```

Or from command line:
```bash
python model_optimizer.py saved_models/stegastamp_pretrained \
    --output_dir optimized_models \
    --precision fp16
```

### C++ API

#### 3. Forensic Watermarker

```cpp
#include "forensic_watermarker.h"

using namespace dci::forensic;

// Configure watermarker
WatermarkConfig config;
config.fmid = 12345;                    // Screen ID
config.resolution = Resolution::DCI_2K;  // 2048x1080
config.fps = 24;
config.pixel_format = PixelFormat::RGB_888;
config.model_path = "/opt/dci/models/forensic_marker_fp16.trt";
config.use_fp16 = true;
config.ring_buffer_size = 8;  // 8 frames for temporal consistency

// Initialize watermarker
ForensicWatermarker watermarker;
WatermarkStatus status = watermarker.Init(config);

if (status != WatermarkStatus::SUCCESS) {
    std::cerr << "Failed to initialize: " 
              << StatusToString(status) << std::endl;
    return -1;
}

// Update timestamp every 15 minutes
uint16_t current_timestamp = calculate_timestamp_from_system_time();
watermarker.UpdateTimestamp(current_timestamp);

// Process frames in real-time
while (running) {
    // Get frame from decryptor
    uint8_t* input_frame = get_next_frame();
    uint8_t* output_frame = allocate_output_buffer();
    
    // Process single frame
    FrameBuffer input, output;
    input.data = input_frame;
    input.width = 2048;
    input.height = 1080;
    input.stride = 2048 * 3;
    input.format = PixelFormat::RGB_888;
    
    output.data = output_frame;
    output.width = 2048;
    output.height = 1080;
    output.stride = 2048 * 3;
    output.format = PixelFormat::RGB_888;
    
    status = watermarker.ProcessFrame(&input, &output);
    
    if (status == WatermarkStatus::SUCCESS) {
        // Send to projector
        send_to_projector(output_frame);
    }
}

// Cleanup
watermarker.Shutdown();
```

#### 4. Batch Processing (GOP)

```cpp
// Process multiple frames at once (more efficient)
const int FRAMES_PER_GOP = 24;  // 1 second @ 24fps
uint8_t* input_buffer = get_gop_buffer();
uint8_t* output_buffer = allocate_gop_buffer();

status = watermarker.ProcessGOP(
    input_buffer,
    output_buffer,
    FRAMES_PER_GOP
);
```

## Performance Targets

### Latency Requirements (DCI CTP)

- **2K @ 24fps**: < 42ms per frame (realtime)
- **2K @ 48fps**: < 21ms per frame (HFR)
- **4K @ 24fps**: < 42ms per frame
- **4K @ 48fps**: < 21ms per frame

### Optimization Strategies

1. **Model Simplification**
   - Remove diffusion: 1000x speedup
   - Single forward pass: ~10-30ms
   - FP16 quantization: 2x speedup, minimal quality loss

2. **Ring Buffer**
   - Pre-allocated GPU memory: Zero allocation overhead
   - Pinned memory: Fast CPU-GPU transfer
   - Batch processing: Amortize overhead

3. **TensorRT Optimization**
   - Layer fusion: Reduce kernel launches
   - INT8 quantization: 4x throughput (requires calibration)
   - Dynamic batching: Process multiple frames together

## Model Training Considerations

### Distillation Approach

Since the original VidStamp uses diffusion (too slow), you need to train a **distilled network**:

```
Teacher Model (Slow)          Student Model (Fast)
┌──────────────────┐         ┌──────────────────┐
│ Diffusion Model  │────────→│ Direct Encoder   │
│ (50+ denoising   │ Distill │ (1 forward pass) │
│  steps)          │         │                  │
└──────────────────┘         └──────────────────┘
    ↓ Output                     ↓ Output
Watermarked Image           Watermarked Image
                                (similar quality)
```

**Training Strategy:**
1. Generate training pairs using teacher model (offline, slow)
2. Train student network to mimic teacher output
3. Loss = MSE(student_output, teacher_output) + perceptual_loss
4. Student learns to embed watermark in single pass

## Security Considerations

### FIPS 140-2 Compliance

- **No plaintext secrets**: FMID and timestamp are not secrets
- **Secure boot**: Model loaded from signed, encrypted storage
- **Tamper detection**: CRC verification on every decode
- **Audit trail**: Log all timestamp updates and errors

### Anti-Piracy Features

- **Temporal redundancy**: Complete payload in every frame
- **Robustness**: Survives compression, cropping, color changes
- **Frame-swap resistance**: Ring buffer enforces temporal consistency
- **Theater identification**: FMID uniquely identifies source

## Testing & Validation

### Unit Tests

```bash
# Test payload encoding/decoding
python -m pytest test_dci_payload.py

# Test timestamp calculations
python -c "from dci_payload import DCIPayload; DCIPayload.test_timestamp_logic()"
```

### Integration Tests

```bash
# Test model optimization pipeline
python model_optimizer.py saved_models/stegastamp_pretrained --output_dir test_output

# Test C++ interface (requires compilation)
make test_forensic_watermarker
./bin/test_forensic_watermarker
```

### Performance Benchmarks

```cpp
// Benchmark single frame processing
auto start = std::chrono::high_resolution_clock::now();
watermarker.ProcessFrame(&input, &output);
auto end = std::chrono::high_resolution_clock::now();
auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
std::cout << "Latency: " << latency.count() / 1000.0 << " ms" << std::endl;
```

## Critical Success Factors

### ✅ Checklist for DCI Certification

- [ ] Every frame contains complete 48-bit payload
- [ ] Timestamp increments every 15 minutes, resets yearly
- [ ] Payload extractable from any 5-minute segment
- [ ] Real-time processing: < 42ms @ 2K/24fps
- [ ] CRC verification success rate > 99.9%
- [ ] Model runs on target hardware (GPU/NPU)
- [ ] Watermark survives JPEG compression (Q=50)
- [ ] Watermark survives 10% luminance variation
- [ ] Zero visual artifacts (LPIPS < 0.1)
- [ ] Secure boot and tamper detection implemented

### ⚠️ Common Pitfalls

1. **Don't split payload across multiple frames**
   - ❌ Wrong: Frame 1 has bits 0-15, Frame 2 has bits 16-31...
   - ✅ Correct: Every frame has all 48 bits

2. **Don't use diffusion in production**
   - ❌ Wrong: Run denoising loop in Media Block
   - ✅ Correct: Use distilled single-pass encoder

3. **Don't use Unix timestamp**
   - ❌ Wrong: Standard Unix epoch timestamp
   - ✅ Correct: DCI timestamp (15-min intervals, yearly reset)

## License

This project extends StegaStamp (MIT License) for DCI compliance. See LICENSE file for details.

## References

- [DCI Compliance Test Plan](https://www.dcimovies.com/specification/index.html)
- [StegaStamp Paper](https://arxiv.org/abs/1904.05343)
- [FIPS 140-2 Security Requirements](https://csrc.nist.gov/publications/detail/fips/140/2/final)

## Support

For DCI integration support, contact the forensic watermarking team.

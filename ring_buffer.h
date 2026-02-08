/**
 * @file ring_buffer.h
 * @brief Ring buffer implementation for temporal video processing
 * 
 * This ring buffer is designed for multi-frame temporal consistency in
 * forensic watermarking. It maintains a sliding window of frames for
 * 3D convolution and temporal attention mechanisms.
 * 
 * Key Features:
 * - Zero-copy design with memory-mapped buffers
 * - Thread-safe for single producer, single consumer
 * - Supports variable frame sizes
 * - GPU-friendly memory layout
 * 
 * @author DCI Integration Team
 * @version 1.0.0
 */

#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <cstdint>
#include <memory>
#include <vector>

namespace dci {
namespace forensic {

/**
 * @enum BufferStatus
 * @brief Status codes for ring buffer operations
 */
enum class BufferStatus {
    SUCCESS = 0,
    ERROR_FULL = 1,
    ERROR_EMPTY = 2,
    ERROR_INVALID_SIZE = 3,
    ERROR_OUT_OF_MEMORY = 4,
    ERROR_INVALID_INDEX = 5
};

/**
 * @struct BufferConfig
 * @brief Configuration for ring buffer
 */
struct BufferConfig {
    uint32_t capacity;          ///< Number of frame slots in buffer
    uint32_t frame_width;       ///< Frame width in pixels
    uint32_t frame_height;      ///< Frame height in pixels
    uint32_t channels;          ///< Number of channels (3 for RGB, 1 for grayscale)
    bool use_gpu_memory;        ///< Allocate in GPU memory (CUDA)
    bool pinned_memory;         ///< Use pinned memory for faster CPU-GPU transfer
};

/**
 * @class RingBuffer
 * @brief Circular buffer for video frames with temporal consistency
 * 
 * This ring buffer maintains a sliding window of frames for temporal
 * processing. It's optimized for the pattern where frames are continuously
 * added and the most recent N frames are processed together.
 * 
 * Memory Layout:
 * The buffer stores frames in a format compatible with neural network
 * inference: [Batch, Channel, Time, Height, Width] or [B, C, T, H, W]
 * 
 * For example, with buffer capacity of 8 frames:
 * - After adding 5 frames: buffer contains frames 0-4
 * - After adding 12 frames: buffer contains frames 4-11 (oldest discarded)
 * 
 * Thread Safety: Single producer, single consumer safe
 * 
 * Usage Example:
 * @code
 * BufferConfig config;
 * config.capacity = 8;
 * config.frame_width = 2048;
 * config.frame_height = 1080;
 * config.channels = 3;
 * config.use_gpu_memory = true;
 * 
 * RingBuffer buffer(config);
 * 
 * // Add frames continuously
 * uint8_t* frame_data = ...;
 * buffer.Push(frame_data);
 * 
 * // Get last N frames for processing
 * float* batch_tensor = buffer.GetBatchTensor(5);  // Get last 5 frames
 * @endcode
 */
class RingBuffer {
public:
    /**
     * @brief Constructor
     * 
     * @param config Buffer configuration
     */
    explicit RingBuffer(const BufferConfig& config);

    /**
     * @brief Destructor
     */
    ~RingBuffer();

    /**
     * @brief Push a new frame into the ring buffer
     * 
     * Adds a frame to the buffer. If the buffer is full, the oldest frame
     * is automatically discarded.
     * 
     * @param frame_data Pointer to frame data (HWC format, uint8)
     * @return BufferStatus::SUCCESS on success, error code otherwise
     */
    BufferStatus Push(const uint8_t* frame_data);

    /**
     * @brief Push a new frame with normalization
     * 
     * Similar to Push, but also normalizes pixel values to [0, 1] range
     * and converts to float32.
     * 
     * @param frame_data Pointer to frame data (HWC format, uint8)
     * @param mean Pointer to mean values for each channel (optional)
     * @param std Pointer to std deviation for each channel (optional)
     * @return BufferStatus::SUCCESS on success, error code otherwise
     */
    BufferStatus PushNormalized(
        const uint8_t* frame_data,
        const float* mean = nullptr,
        const float* std = nullptr
    );

    /**
     * @brief Get a batch tensor of the most recent N frames
     * 
     * Returns a pointer to a tensor containing the last N frames in
     * neural network format: [1, C, N, H, W]
     * 
     * If fewer than N frames are available, returns what's available.
     * 
     * @param num_frames Number of frames to include (must be <= capacity)
     * @param output_size Output: Actual number of frames returned
     * @return Pointer to batch tensor (float32), or nullptr on error
     */
    float* GetBatchTensor(uint32_t num_frames, uint32_t* output_size = nullptr);

    /**
     * @brief Get raw pointer to a specific frame
     * 
     * @param index Frame index (0 = oldest, size-1 = newest)
     * @return Pointer to frame data, or nullptr if invalid index
     */
    const uint8_t* GetFrame(uint32_t index) const;

    /**
     * @brief Get the current number of frames in buffer
     * 
     * @return Number of frames currently stored
     */
    uint32_t Size() const;

    /**
     * @brief Get the buffer capacity
     * 
     * @return Maximum number of frames the buffer can hold
     */
    uint32_t Capacity() const;

    /**
     * @brief Check if buffer is full
     * 
     * @return true if buffer is at capacity
     */
    bool IsFull() const;

    /**
     * @brief Check if buffer is empty
     * 
     * @return true if buffer contains no frames
     */
    bool IsEmpty() const;

    /**
     * @brief Clear all frames from buffer
     */
    void Clear();

    /**
     * @brief Get pointer to GPU memory (if allocated)
     * 
     * @return CUDA device pointer, or nullptr if using CPU memory
     */
    void* GetDevicePointer() const;

    /**
     * @brief Copy data to external buffer
     * 
     * Copies the batch tensor to an external buffer. Useful for
     * interfacing with inference engines.
     * 
     * @param dest Destination buffer
     * @param num_frames Number of frames to copy
     * @param dest_size Size of destination buffer in bytes
     * @return BufferStatus::SUCCESS on success, error code otherwise
     */
    BufferStatus CopyToBuffer(
        void* dest,
        uint32_t num_frames,
        size_t dest_size
    ) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Disable copy and assignment
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;
};

/**
 * @brief Helper function to calculate required buffer size
 * 
 * @param width Frame width
 * @param height Frame height
 * @param channels Number of channels
 * @param num_frames Number of frames
 * @return Required buffer size in bytes
 */
inline size_t CalculateBufferSize(
    uint32_t width,
    uint32_t height,
    uint32_t channels,
    uint32_t num_frames
) {
    return static_cast<size_t>(width) * height * channels * num_frames;
}

} // namespace forensic
} // namespace dci

#endif // RING_BUFFER_H

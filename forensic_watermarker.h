/**
 * @file forensic_watermarker.h
 * @brief DCI-compliant Forensic Watermarking Interface for Digital Cinema Media Block
 * 
 * This header defines the C++ interface for embedding forensic marks into
 * digital cinema content streams. Designed for integration with FIPS 140-2
 * Level 3 secure media blocks.
 * 
 * Key Features:
 * - Real-time watermark embedding for 2K/4K @ 24fps/48fps
 * - Temporal consistency with ring buffer architecture
 * - Zero-copy design for low latency
 * - DCI CTP compliant payload structure
 * 
 * @author DCI Integration Team
 * @version 1.0.0
 */

#ifndef FORENSIC_WATERMARKER_H
#define FORENSIC_WATERMARKER_H

#include <cstdint>
#include <memory>
#include <string>

namespace dci {
namespace forensic {

/**
 * @enum WatermarkStatus
 * @brief Status codes for watermarking operations
 */
enum class WatermarkStatus {
    SUCCESS = 0,
    ERROR_INIT_FAILED = 1,
    ERROR_INVALID_FMID = 2,
    ERROR_INVALID_TIMESTAMP = 3,
    ERROR_BUFFER_SIZE_MISMATCH = 4,
    ERROR_MODEL_LOAD_FAILED = 5,
    ERROR_INFERENCE_FAILED = 6,
    ERROR_NULL_POINTER = 7,
    ERROR_UNSUPPORTED_FORMAT = 8
};

/**
 * @enum PixelFormat
 * @brief Supported pixel formats for input/output buffers
 */
enum class PixelFormat {
    RGB_888,     ///< 8-bit RGB (24 bits per pixel)
    RGBA_8888,   ///< 8-bit RGBA (32 bits per pixel)
    YUV_420,     ///< YUV 4:2:0 planar
    YUV_422      ///< YUV 4:2:2 planar
};

/**
 * @enum Resolution
 * @brief Standard DCI resolutions
 */
enum class Resolution {
    DCI_2K,      ///< 2048x1080
    DCI_4K,      ///< 4096x2160
    CUSTOM       ///< Custom resolution
};

/**
 * @struct WatermarkConfig
 * @brief Configuration for watermark initialization
 */
struct WatermarkConfig {
    uint32_t fmid;                    ///< Forensic Marking ID (20-bit, 0-1048575)
    uint16_t initial_timestamp;       ///< Initial timestamp (0-65535)
    Resolution resolution;            ///< Video resolution
    uint32_t width;                   ///< Frame width (for CUSTOM resolution)
    uint32_t height;                  ///< Frame height (for CUSTOM resolution)
    PixelFormat pixel_format;         ///< Pixel format
    uint32_t fps;                     ///< Frames per second (24, 48, etc.)
    std::string model_path;           ///< Path to TensorRT/ONNX model
    bool use_fp16;                    ///< Use FP16 precision for inference
    uint32_t ring_buffer_size;        ///< Number of frames in ring buffer (default: 8)
};

/**
 * @struct FrameBuffer
 * @brief Frame buffer descriptor for zero-copy operations
 */
struct FrameBuffer {
    uint8_t* data;                    ///< Pointer to frame data
    uint32_t width;                   ///< Frame width
    uint32_t height;                  ///< Frame height
    uint32_t stride;                  ///< Row stride in bytes
    PixelFormat format;               ///< Pixel format
    uint64_t timestamp_ns;            ///< Frame timestamp in nanoseconds
};

/**
 * @class ForensicWatermarker
 * @brief Main interface for forensic watermarking in digital cinema systems
 * 
 * This class provides the core functionality for embedding DCI-compliant
 * forensic marks into video streams in real-time.
 * 
 * Thread Safety: This class is NOT thread-safe. Create separate instances
 * for concurrent processing pipelines.
 * 
 * Usage Example:
 * @code
 * WatermarkConfig config;
 * config.fmid = 12345;
 * config.resolution = Resolution::DCI_2K;
 * config.fps = 24;
 * config.model_path = "/opt/dci/models/forensic_marker.trt";
 * config.use_fp16 = true;
 * 
 * ForensicWatermarker watermarker;
 * if (watermarker.Init(config) != WatermarkStatus::SUCCESS) {
 *     // Handle error
 * }
 * 
 * // Update timestamp every 15 minutes
 * watermarker.UpdateTimestamp(new_timestamp);
 * 
 * // Process frames
 * FrameBuffer input, output;
 * // ... setup buffers ...
 * watermarker.ProcessFrame(&input, &output);
 * @endcode
 */
class ForensicWatermarker {
public:
    /**
     * @brief Constructor
     */
    ForensicWatermarker();

    /**
     * @brief Destructor
     */
    ~ForensicWatermarker();

    /**
     * @brief Initialize the watermarker with configuration
     * 
     * This method loads the neural network model, initializes the ring buffer,
     * and prepares all resources for watermark embedding.
     * 
     * @param config Configuration structure
     * @return WatermarkStatus::SUCCESS on success, error code otherwise
     */
    WatermarkStatus Init(const WatermarkConfig& config);

    /**
     * @brief Initialize with FMID only (legacy interface)
     * 
     * Uses default configuration for DCI 2K @ 24fps
     * 
     * @param fmid Forensic Marking ID (0-1048575)
     * @return WatermarkStatus::SUCCESS on success, error code otherwise
     */
    WatermarkStatus Init(uint32_t fmid);

    /**
     * @brief Update the timestamp component of the payload
     * 
     * Should be called every 15 minutes to maintain accurate forensic marks.
     * Can be called at any time; the new timestamp will be used for subsequent
     * frames.
     * 
     * @param time_code 16-bit timestamp (0-65535)
     * @return WatermarkStatus::SUCCESS on success, error code otherwise
     */
    WatermarkStatus UpdateTimestamp(uint16_t time_code);

    /**
     * @brief Process a single frame (zero-copy when possible)
     * 
     * Embeds the forensic watermark into a single frame. The frame is added
     * to the ring buffer for temporal consistency.
     * 
     * @param input_buffer Pointer to input frame buffer
     * @param output_buffer Pointer to output frame buffer
     * @return WatermarkStatus::SUCCESS on success, error code otherwise
     */
    WatermarkStatus ProcessFrame(
        const FrameBuffer* input_buffer,
        FrameBuffer* output_buffer
    );

    /**
     * @brief Process a Group of Pictures (GOP)
     * 
     * Processes multiple frames with temporal consistency. This is the
     * recommended method for batch processing.
     * 
     * @param input_buffer Pointer to input buffer (contiguous frames)
     * @param output_buffer Pointer to output buffer (contiguous frames)
     * @param frame_count Number of frames to process
     * @return WatermarkStatus::SUCCESS on success, error code otherwise
     */
    WatermarkStatus ProcessGOP(
        uint8_t* input_buffer,
        uint8_t* output_buffer,
        int frame_count
    );

    /**
     * @brief Get current watermark statistics
     * 
     * @param frames_processed Output: Total frames processed
     * @param avg_latency_ms Output: Average processing latency in milliseconds
     * @return WatermarkStatus::SUCCESS on success, error code otherwise
     */
    WatermarkStatus GetStats(
        uint64_t* frames_processed,
        double* avg_latency_ms
    ) const;

    /**
     * @brief Reset the watermarker state
     * 
     * Clears the ring buffer and resets statistics. Does not reload the model.
     * 
     * @return WatermarkStatus::SUCCESS on success, error code otherwise
     */
    WatermarkStatus Reset();

    /**
     * @brief Shutdown and release all resources
     * 
     * Releases GPU memory, closes model, and cleans up ring buffer.
     */
    void Shutdown();

    /**
     * @brief Check if watermarker is initialized
     * 
     * @return true if initialized, false otherwise
     */
    bool IsInitialized() const;

    /**
     * @brief Get version information
     * 
     * @return Version string (e.g., "1.0.0")
     */
    static std::string GetVersion();

private:
    class Impl;  // Forward declaration for PIMPL pattern
    std::unique_ptr<Impl> impl_;  // Private implementation
    
    // Disable copy and assignment
    ForensicWatermarker(const ForensicWatermarker&) = delete;
    ForensicWatermarker& operator=(const ForensicWatermarker&) = delete;
};

/**
 * @brief Convert status code to string description
 * 
 * @param status Status code
 * @return Human-readable string description
 */
const char* StatusToString(WatermarkStatus status);

} // namespace forensic
} // namespace dci

#endif // FORENSIC_WATERMARKER_H

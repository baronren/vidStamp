"""
Example implementation demonstrating DCI-compliant forensic watermarking.

This script shows how to integrate the payload encoder with the existing
StegaStamp model for real-time watermark embedding.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import time
from typing import List, Tuple

from dci_payload import DCIPayload


class ForensicWatermarkEmbedder:
    """
    High-level interface for embedding DCI-compliant forensic watermarks.
    
    This class combines:
    - DCI payload encoding (timestamp + FMID + CRC)
    - StegaStamp encoder for image watermarking
    - Frame batching for temporal consistency
    """
    
    def __init__(
        self,
        model_path: str,
        fmid: int,
        image_size: Tuple[int, int] = (400, 400)
    ):
        """
        Initialize the embedder.
        
        Args:
            model_path: Path to TensorFlow saved model
            fmid: Forensic Marking ID (0-1048575)
            image_size: Image dimensions (width, height)
        """
        self.fmid = fmid
        self.image_size = image_size
        self.payload_encoder = DCIPayload(fmid)
        
        # Load TensorFlow model
        self.session = tf.InteractiveSession(graph=tf.Graph())
        model = tf.saved_model.loader.load(
            self.session,
            [tf.saved_model.tag_constants.SERVING],
            model_path
        )
        
        # Get tensor references
        signature = model.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.input_secret = self.session.graph.get_tensor_by_name(
            signature.inputs['secret'].name
        )
        self.input_image = self.session.graph.get_tensor_by_name(
            signature.inputs['image'].name
        )
        self.output_stegastamp = self.session.graph.get_tensor_by_name(
            signature.outputs['stegastamp'].name
        )
        self.output_residual = self.session.graph.get_tensor_by_name(
            signature.outputs['residual'].name
        )
        
        print(f"ForensicWatermarkEmbedder initialized with FMID={fmid}")
    
    def update_timestamp(self, timestamp: int = None):
        """
        Update the current timestamp.
        
        Should be called every 15 minutes in production.
        
        Args:
            timestamp: 16-bit timestamp (auto-calculated if None)
        """
        if timestamp is None:
            timestamp = DCIPayload.calculate_timestamp()
        self.current_timestamp = timestamp
        print(f"Timestamp updated to {timestamp}")
    
    def embed_frame(
        self,
        image: np.ndarray,
        timestamp: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed forensic watermark into a single frame.
        
        Args:
            image: Input image as numpy array (H, W, 3), float32, range [0, 1]
            timestamp: Optional timestamp override
            
        Returns:
            Tuple of (watermarked_image, residual)
        """
        # Generate 48-bit payload
        if timestamp is None:
            timestamp = getattr(self, 'current_timestamp', None)
        
        payload_bits = self.payload_encoder.encode_for_vidstamp(timestamp)
        
        # Prepare image
        if image.shape[:2] != self.image_size:
            # Resize if needed
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
            img_pil = img_pil.resize(self.image_size, Image.LANCZOS)
            image = np.array(img_pil, dtype=np.float32) / 255.0
        
        # Run inference
        feed_dict = {
            self.input_secret: [payload_bits],
            self.input_image: [image]
        }
        
        watermarked, residual = self.session.run(
            [self.output_stegastamp, self.output_residual],
            feed_dict=feed_dict
        )
        
        return watermarked[0], residual[0]
    
    def embed_batch(
        self,
        images: List[np.ndarray],
        timestamp: int = None
    ) -> List[np.ndarray]:
        """
        Embed forensic watermark into a batch of frames.
        
        All frames in the batch get the same payload (same timestamp).
        For temporal consistency across a video segment.
        
        Args:
            images: List of input images
            timestamp: Optional timestamp override
            
        Returns:
            List of watermarked images
        """
        watermarked_images = []
        
        for image in images:
            watermarked, _ = self.embed_frame(image, timestamp)
            watermarked_images.append(watermarked)
        
        return watermarked_images
    
    def embed_video_segment(
        self,
        video_frames: List[np.ndarray],
        output_dir: str,
        segment_name: str = "segment"
    ):
        """
        Embed forensic watermark into a video segment and save frames.
        
        Args:
            video_frames: List of video frames
            output_dir: Directory to save watermarked frames
            segment_name: Prefix for output filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Update timestamp for this segment
        self.update_timestamp()
        
        print(f"Embedding watermark into {len(video_frames)} frames...")
        start_time = time.time()
        
        for i, frame in enumerate(video_frames):
            watermarked, residual = self.embed_frame(frame)
            
            # Save watermarked frame
            img_wm = Image.fromarray((watermarked * 255).astype(np.uint8))
            img_wm.save(os.path.join(output_dir, f"{segment_name}_frame_{i:04d}.png"))
            
            # Optionally save residual for debugging
            residual_vis = ((residual + 0.5) * 255).astype(np.uint8)
            img_res = Image.fromarray(residual_vis)
            img_res.save(os.path.join(output_dir, f"{segment_name}_residual_{i:04d}.png"))
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed
                print(f"  Processed {i+1}/{len(video_frames)} frames ({fps:.2f} fps)")
        
        total_time = time.time() - start_time
        avg_fps = len(video_frames) / total_time
        print(f"Completed in {total_time:.2f}s ({avg_fps:.2f} fps)")
    
    def benchmark_performance(self, num_frames: int = 100):
        """
        Benchmark the embedding performance.
        
        Args:
            num_frames: Number of frames to process for benchmark
        """
        print(f"\nBenchmarking with {num_frames} frames...")
        
        # Create dummy frame
        dummy_frame = np.random.rand(
            self.image_size[1],
            self.image_size[0],
            3
        ).astype(np.float32)
        
        # Warmup
        for _ in range(5):
            self.embed_frame(dummy_frame)
        
        # Benchmark
        latencies = []
        start_time = time.time()
        
        for _ in range(num_frames):
            frame_start = time.time()
            watermarked, _ = self.embed_frame(dummy_frame)
            frame_end = time.time()
            latencies.append((frame_end - frame_start) * 1000)  # ms
        
        total_time = time.time() - start_time
        
        # Statistics
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        avg_fps = num_frames / total_time
        
        print(f"\nPerformance Metrics:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Average latency: {avg_latency:.2f} ms")
        print(f"  P50 latency: {p50_latency:.2f} ms")
        print(f"  P95 latency: {p95_latency:.2f} ms")
        print(f"  P99 latency: {p99_latency:.2f} ms")
        
        # Check DCI requirements
        dci_target_2k_24fps = 42  # ms
        dci_target_2k_48fps = 21  # ms
        
        print(f"\nDCI Compliance Check:")
        print(f"  2K @ 24fps (<42ms): {'✓ PASS' if p99_latency < dci_target_2k_24fps else '✗ FAIL'}")
        print(f"  2K @ 48fps (<21ms): {'✓ PASS' if p99_latency < dci_target_2k_48fps else '✗ FAIL'}")


def demo_single_image(model_path: str, image_path: str, output_dir: str):
    """
    Demonstrate watermarking a single image.
    
    Args:
        model_path: Path to TensorFlow model
        image_path: Path to input image
        output_dir: Directory to save output
    """
    print("="*60)
    print("DCI Forensic Watermarking Demo - Single Image")
    print("="*60)
    
    # Initialize embedder
    fmid = 12345  # Screen ID
    embedder = ForensicWatermarkEmbedder(model_path, fmid)
    embedder.update_timestamp()
    
    # Load image
    print(f"\nLoading image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    img = img.resize((400, 400), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Embed watermark
    print("Embedding forensic watermark...")
    watermarked, residual = embedder.embed_frame(img_array)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    img_wm = Image.fromarray((watermarked * 255).astype(np.uint8))
    img_wm.save(os.path.join(output_dir, 'watermarked.png'))
    
    residual_vis = ((residual + 0.5) * 255).astype(np.uint8)
    img_res = Image.fromarray(residual_vis)
    img_res.save(os.path.join(output_dir, 'residual.png'))
    
    print(f"Results saved to {output_dir}")
    print("  - watermarked.png: Image with embedded watermark")
    print("  - residual.png: Visualization of watermark signal")
    
    # Display payload info
    payload_bits = embedder.payload_encoder.encode_payload()
    timestamp, fmid_decoded, crc, is_valid = DCIPayload.decode_payload(payload_bits)
    
    print(f"\nEmbedded Payload:")
    print(f"  FMID: {fmid_decoded}")
    print(f"  Timestamp: {timestamp}")
    print(f"  CRC: {crc:03X}")
    print(f"  Valid: {is_valid}")
    
    day, hour, minute = DCIPayload.timestamp_to_datetime(timestamp)
    print(f"  Time: Day {day} at {hour:02d}:{minute:02d}")


def demo_benchmark(model_path: str):
    """
    Demonstrate performance benchmarking.
    
    Args:
        model_path: Path to TensorFlow model
    """
    print("="*60)
    print("DCI Forensic Watermarking Demo - Performance Benchmark")
    print("="*60)
    
    fmid = 12345
    embedder = ForensicWatermarkEmbedder(model_path, fmid)
    embedder.update_timestamp()
    
    embedder.benchmark_performance(num_frames=100)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DCI Forensic Watermarking Demo"
    )
    parser.add_argument(
        'model',
        type=str,
        help='Path to TensorFlow saved model'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['image', 'benchmark'],
        default='benchmark',
        help='Demo mode'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Input image path (for image mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'image':
        if not args.image:
            print("Error: --image required for image mode")
            exit(1)
        demo_single_image(args.model, args.image, args.output)
    else:
        demo_benchmark(args.model)

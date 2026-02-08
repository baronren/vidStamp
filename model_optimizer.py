"""
Model Optimization and Export Utilities for DCI Forensic Watermarking

This module provides tools to convert the VidStamp/StegaStamp TensorFlow model
to production-ready formats (TensorRT, ONNX) with quantization support.

Key Features:
- Remove diffusion dependencies
- Export encoder-decoder to ONNX
- Apply FP16/INT8 quantization
- Generate TensorRT engine for embedded deployment
"""

import tensorflow as tf
import numpy as np
import os
from typing import Tuple, Optional


class ModelOptimizer:
    """
    Optimizer for converting StegaStamp/VidStamp models to real-time inference.
    
    The original model is too slow for real-time use due to its complex
    encoder-decoder architecture and spatial transformer network. This
    optimizer creates a lightweight version suitable for embedded deployment.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize model optimizer.
        
        Args:
            model_path: Path to saved TensorFlow model
        """
        self.model_path = model_path
        self.session = None
        self.graph = None
        
    def load_model(self) -> bool:
        """
        Load the TensorFlow saved model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.session = tf.Session(graph=tf.Graph())
            tf.saved_model.loader.load(
                self.session,
                [tf.saved_model.tag_constants.SERVING],
                self.model_path
            )
            self.graph = self.session.graph
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_encoder_only(self, output_path: str) -> bool:
        """
        Extract only the encoder portion of the model.
        
        This removes the decoder and discriminator, keeping only the
        watermark embedding encoder. This is what we need for the
        Media Block integration.
        
        Args:
            output_path: Path to save the extracted encoder
            
        Returns:
            True if successful, False otherwise
        """
        if self.session is None or self.graph is None:
            print("Model not loaded. Call load_model() first.")
            return False
        
        try:
            # Get input tensors
            input_secret = self.graph.get_tensor_by_name('secret:0')
            input_image = self.graph.get_tensor_by_name('image:0')
            
            # Get encoder output (residual or stegastamp)
            output_stegastamp = self.graph.get_tensor_by_name('stegastamp:0')
            output_residual = self.graph.get_tensor_by_name('residual:0')
            
            # Create new saved model with only encoder
            builder = tf.saved_model.builder.SavedModelBuilder(output_path)
            
            # Define signature
            tensor_info_secret = tf.saved_model.utils.build_tensor_info(input_secret)
            tensor_info_image = tf.saved_model.utils.build_tensor_info(input_image)
            tensor_info_stegastamp = tf.saved_model.utils.build_tensor_info(output_stegastamp)
            tensor_info_residual = tf.saved_model.utils.build_tensor_info(output_residual)
            
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        'secret': tensor_info_secret,
                        'image': tensor_info_image
                    },
                    outputs={
                        'stegastamp': tensor_info_stegastamp,
                        'residual': tensor_info_residual
                    },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )
            
            builder.add_meta_graph_and_variables(
                self.session,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        prediction_signature
                }
            )
            
            builder.save()
            print(f"Encoder-only model saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error extracting encoder: {e}")
            return False
    
    def export_to_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, int, int] = (400, 400, 3),
        secret_size: int = 100
    ) -> bool:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            input_shape: Input image shape (H, W, C)
            secret_size: Size of secret input
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import tf2onnx
            
            if self.session is None or self.graph is None:
                print("Model not loaded. Call load_model() first.")
                return False
            
            # Define input/output nodes
            input_names = ['secret:0', 'image:0']
            output_names = ['stegastamp:0']
            
            # Convert to ONNX
            onnx_model, _ = tf2onnx.convert.from_session(
                self.session,
                input_names=input_names,
                output_names=output_names
            )
            
            # Save ONNX model
            with open(output_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"ONNX model saved to {output_path}")
            return True
            
        except ImportError:
            print("tf2onnx not installed. Install with: pip install tf2onnx")
            return False
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")
            return False
    
    def create_tensorrt_engine(
        self,
        onnx_path: str,
        output_path: str,
        precision: str = 'fp16',
        max_batch_size: int = 1
    ) -> bool:
        """
        Create TensorRT engine from ONNX model.
        
        This requires TensorRT to be installed and is typically run on
        the target embedded system.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TensorRT engine
            precision: Precision mode ('fp32', 'fp16', or 'int8')
            max_batch_size: Maximum batch size
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import tensorrt as trt
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            
            if precision == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
                print("Using FP16 precision")
            elif precision == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)
                print("Using INT8 precision (requires calibration data)")
            
            # Build engine
            print("Building TensorRT engine... This may take a while.")
            engine = builder.build_engine(network, config)
            
            if engine is None:
                print("ERROR: Failed to build TensorRT engine")
                return False
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            print(f"TensorRT engine saved to {output_path}")
            return True
            
        except ImportError:
            print("TensorRT not installed. Install TensorRT Python API.")
            return False
        except Exception as e:
            print(f"Error creating TensorRT engine: {e}")
            return False


def create_optimized_pipeline(
    model_path: str,
    output_dir: str,
    precision: str = 'fp16'
) -> bool:
    """
    Complete optimization pipeline: TF -> Encoder-only -> ONNX -> TensorRT
    
    Args:
        model_path: Path to original TensorFlow model
        output_dir: Directory to save optimized models
        precision: Precision for TensorRT ('fp32', 'fp16', 'int8')
        
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(output_dir, exist_ok=True)
    
    optimizer = ModelOptimizer(model_path)
    
    # Step 1: Load model
    print("Step 1: Loading TensorFlow model...")
    if not optimizer.load_model():
        return False
    
    # Step 2: Extract encoder
    print("\nStep 2: Extracting encoder-only model...")
    encoder_path = os.path.join(output_dir, 'encoder_only')
    if not optimizer.extract_encoder_only(encoder_path):
        return False
    
    # Step 3: Export to ONNX
    print("\nStep 3: Exporting to ONNX...")
    onnx_path = os.path.join(output_dir, 'forensic_marker.onnx')
    if not optimizer.export_to_onnx(onnx_path):
        print("ONNX export failed, but encoder extraction succeeded.")
        print("You can manually export to ONNX using tf2onnx tool.")
    
    # Step 4: Create TensorRT engine (optional, requires TensorRT)
    print("\nStep 4: Creating TensorRT engine...")
    trt_path = os.path.join(output_dir, f'forensic_marker_{precision}.trt')
    if os.path.exists(onnx_path):
        if not optimizer.create_tensorrt_engine(onnx_path, trt_path, precision):
            print("TensorRT conversion failed.")
            print("You can manually convert on target system with TensorRT installed.")
    
    print("\n" + "="*50)
    print("Optimization pipeline completed!")
    print(f"Output directory: {output_dir}")
    print("="*50)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimize StegaStamp model for real-time DCI deployment"
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to original TensorFlow saved model'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./optimized_models',
        help='Directory to save optimized models'
    )
    parser.add_argument(
        '--precision',
        type=str,
        choices=['fp32', 'fp16', 'int8'],
        default='fp16',
        help='TensorRT precision mode'
    )
    
    args = parser.parse_args()
    
    success = create_optimized_pipeline(
        args.model_path,
        args.output_dir,
        args.precision
    )
    
    exit(0 if success else 1)

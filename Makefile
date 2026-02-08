# Makefile for DCI Forensic Watermarking System
#
# This Makefile provides build targets for the C++ components.
# Adjust paths and flags based on your system configuration.

CXX = g++
CXXFLAGS = -std=c++14 -Wall -Wextra -O3
INCLUDES = -I. -I/usr/local/cuda/include
LDFLAGS = -L/usr/local/cuda/lib64
LIBS = -lcudart

# Optional: TensorRT support
# INCLUDES += -I/usr/include/x86_64-linux-gnu
# LIBS += -lnvinfer -lnvinfer_plugin -lnvonnxparser

# Optional: ONNX Runtime support
# INCLUDES += -I/usr/local/include/onnxruntime
# LIBS += -lonnxruntime

# Targets
EXAMPLE = forensic_watermarker_example
TEST_TARGETS = test_payload test_buffer

.PHONY: all clean example tests help

all: example

example: $(EXAMPLE)

$(EXAMPLE): forensic_watermarker_example.cpp forensic_watermarker.h ring_buffer.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXAMPLE) forensic_watermarker_example.cpp $(LDFLAGS) $(LIBS)
	@echo "Built $(EXAMPLE) successfully!"
	@echo "Run with: ./$(EXAMPLE)"

# Clean build artifacts
clean:
	rm -f $(EXAMPLE) $(TEST_TARGETS)
	rm -rf *.o *.dSYM
	@echo "Cleaned build artifacts"

# Help target
help:
	@echo "DCI Forensic Watermarking System - Build Instructions"
	@echo "======================================================"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build example application"
	@echo "  make example      - Build forensic watermarker example"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - CUDA Toolkit (for GPU support)"
	@echo "  - TensorRT (optional, for inference)"
	@echo "  - ONNX Runtime (optional, for inference)"
	@echo ""
	@echo "Configuration:"
	@echo "  Edit INCLUDES and LIBS variables in Makefile to match your system"
	@echo ""
	@echo "Note: The example is a reference implementation demonstrating"
	@echo "      the interface. Production deployment requires:"
	@echo "      - Trained model (TensorRT or ONNX)"
	@echo "      - Inference engine integration"
	@echo "      - Hardware-specific optimization"

# Installation target (optional)
install:
	@echo "Installation not implemented. This is a reference implementation."
	@echo "For production deployment:"
	@echo "  1. Build optimized model (see model_optimizer.py)"
	@echo "  2. Integrate with your Media Block software"
	@echo "  3. Deploy to secure embedded system"
	@echo "  4. Perform DCI CTP certification testing"

.PHONY: install

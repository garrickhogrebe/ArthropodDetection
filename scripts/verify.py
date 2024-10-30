"""
verify.py

Simple test script to verify that TensorFlow is correctly installed and has access to the GPU, if available.
Run this script to quickly confirm TensorFlow and GPU configuration in your environment.

Usage:
    Run this script directly:
        python verify.py
"""

import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check if GPU is available
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("TensorFlow has access to the GPU")
    print("GPU device:", tf.test.gpu_device_name())
else:
    print("TensorFlow is running on the CPU")


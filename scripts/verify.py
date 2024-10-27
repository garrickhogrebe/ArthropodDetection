"""
Simple test script to verify tensorflow is instaleld and connected to the gpu
"""

import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check if GPU is available
if tf.test.is_gpu_available():
    print("TensorFlow has access to the GPU")
    print("GPU device:", tf.test.gpu_device_name())
else:
    print("TensorFlow is running on the CPU")

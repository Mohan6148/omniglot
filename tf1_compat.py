#!/usr/bin/env python3
"""
TensorFlow 1.x Compatibility Layer for Google Colab
This script enables TensorFlow 1.x code to run in TensorFlow 2.x environment
"""

import tensorflow as tf
import warnings

# Disable TensorFlow 2.x behavior and enable TensorFlow 1.x compatibility mode
print("Enabling TensorFlow 1.x compatibility mode...")
print(f"TensorFlow version: {tf.__version__}")

# Disable eager execution to use graph mode (TF 1.x style)
if tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    print("✓ TensorFlow 1.x compatibility mode enabled")
    print("✓ Eager execution disabled")
    print("✓ Graph mode enabled")
else:
    print("✓ TensorFlow 1.x detected, no compatibility layer needed")

# Suppress warnings
warnings.filterwarnings('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)

print("\nYou can now run the original classifier7.py code!")

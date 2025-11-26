#!/usr/bin/env python3
"""
Evaluate Custom 50-Epoch Model
Test accuracy on FER2013 test set
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to model and test data
test_data_dir = 'data/fer2013/test'
model_path = 'models/emotion_model_50epochs.h5'

# Image parameters
img_size = (48, 48)
batch_size = 64

print("🎭 EVALUATING 50-EPOCH MODEL")
print("=" * 60)

# Check if model exists
if not os.path.exists(model_path):
    print(f"❌ Model not found at {model_path}")
    sys.exit(1)

print(f"📁 Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path, compile=False)

# Recompile model for evaluation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("✅ Model loaded and compiled\n")

# Prepare test data generator
print(f"📊 Preparing test data from {test_data_dir}...")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

print(f"✅ Test samples: {test_generator.samples}\n")

# Evaluate
print("🚀 Evaluating model on test set...")
print("=" * 60)
test_loss, test_accuracy = model.evaluate(test_generator)

print("\n📊 RESULTS")
print("=" * 60)
print(f"✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print("=" * 60)

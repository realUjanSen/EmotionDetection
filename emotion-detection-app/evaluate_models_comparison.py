#!/usr/bin/env python3
"""
Evaluate both models: 100 Epochs Custom Model vs XCEPTION
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Test data path
test_dir = os.path.join('data', 'fer2013', 'test')

# Load test data
print("\n📊 Loading test data...")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

print(f"✅ Test samples: {test_generator.samples}")

# ============= Evaluate Custom 100 Epochs Model =============
print("\n" + "="*60)
print("🎯 EVALUATING CUSTOM CNN (100 EPOCHS MODEL)")
print("="*60)

try:
    custom_model = load_model('models/emotion_model_100epochs.h5')
    print("✅ Loaded: emotion_model_100epochs.h5")
    
    # Reset test generator
    test_generator.reset()
    
    # Evaluate
    test_loss, test_accuracy = custom_model.evaluate(test_generator, verbose=0)
    print(f"\n📊 Custom CNN (100 Epochs) Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    custom_accuracy = test_accuracy
    
except Exception as e:
    print(f"❌ Error loading custom model: {e}")
    custom_accuracy = None

# ============= Evaluate XCEPTION Model =============
print("\n" + "="*60)
print("🎯 EVALUATING XCEPTION MODEL")
print("="*60)

try:
    xception_model = load_model('models/fer2013_big_XCEPTION.54-0.66.hdf5')
    print("✅ Loaded: fer2013_big_XCEPTION.54-0.66.hdf5")
    
    # Reset test generator
    test_generator.reset()
    
    # Evaluate
    test_loss, test_accuracy = xception_model.evaluate(test_generator, verbose=0)
    print(f"\n📊 XCEPTION Model Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    xception_accuracy = test_accuracy
    
except Exception as e:
    print(f"❌ Error loading XCEPTION model: {e}")
    xception_accuracy = None

# ============= Comparison =============
print("\n" + "="*60)
print("📈 COMPARISON")
print("="*60)

if custom_accuracy and xception_accuracy:
    improvement = custom_accuracy - xception_accuracy
    ratio = custom_accuracy / xception_accuracy if xception_accuracy > 0 else 0
    
    print(f"\nCustom CNN (100 Epochs): {custom_accuracy*100:.2f}%")
    print(f"XCEPTION:               {xception_accuracy*100:.2f}%")
    print(f"\nImprovement: +{improvement*100:.2f} percentage points")
    print(f"Ratio: {ratio:.2f}× better")
    
print("\n" + "="*60)

#!/usr/bin/env python3
"""
Evaluate both models with correct dimensions
Custom CNN: 48x48 grayscale
XCEPTION: 64x64 (may need RGB)
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Test data path
test_dir = os.path.join('data', 'fer2013', 'test')

# ============= Evaluate Custom 48x48 Model =============
print("\n" + "="*70)
print("🎯 EVALUATING CUSTOM CNN (100 EPOCHS, 48×48 GRAYSCALE)")
print("="*70)

try:
    print("📊 Loading test data (48×48 grayscale)...")
    test_datagen_48 = ImageDataGenerator(rescale=1./255)
    test_generator_48 = test_datagen_48.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✅ Test samples: {test_generator_48.samples}")
    
    custom_model = load_model('models/emotion_model_100epochs.h5')
    print("✅ Loaded: emotion_model_100epochs.h5")
    
    # Evaluate
    test_loss, test_accuracy = custom_model.evaluate(test_generator_48, verbose=0)
    print(f"\n📊 Custom CNN Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    custom_accuracy = test_accuracy
    custom_loss = test_loss
    
except Exception as e:
    print(f"❌ Error: {e}")
    custom_accuracy = None
    custom_loss = None

# ============= Evaluate XCEPTION 64x64 Model =============
print("\n" + "="*70)
print("🎯 EVALUATING XCEPTION (64×64, MAY USE RGB)")
print("="*70)

try:
    print("📊 Loading test data (64×64 RGB)...")
    test_datagen_64 = ImageDataGenerator(rescale=1./255)
    test_generator_64 = test_datagen_64.flow_from_directory(
        test_dir,
        target_size=(64, 64),
        batch_size=64,
        color_mode='rgb',  # Try RGB first
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✅ Test samples: {test_generator_64.samples}")
    
    # Try loading XCEPTION with custom optimizer to avoid compatibility issues
    print("✅ Loading: fer2013_big_XCEPTION.54-0.66.hdf5")
    xception_model = load_model(
        'models/fer2013_big_XCEPTION.54-0.66.hdf5',
        compile=False
    )
    
    # Recompile with current Keras version
    xception_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model compiled successfully")
    
    # Evaluate
    test_loss, test_accuracy = xception_model.evaluate(test_generator_64, verbose=0)
    print(f"\n📊 XCEPTION Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    xception_accuracy = test_accuracy
    xception_loss = test_loss
    
except Exception as e:
    print(f"❌ Error with RGB: {e}")
    print("\n📊 Trying with grayscale instead...")
    
    try:
        test_datagen_64_gray = ImageDataGenerator(rescale=1./255)
        test_generator_64_gray = test_datagen_64_gray.flow_from_directory(
            test_dir,
            target_size=(64, 64),
            batch_size=64,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✅ Test samples: {test_generator_64_gray.samples}")
        
        xception_model = load_model(
            'models/fer2013_big_XCEPTION.54-0.66.hdf5',
            compile=False
        )
        
        xception_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        test_loss, test_accuracy = xception_model.evaluate(test_generator_64_gray, verbose=0)
        print(f"\n📊 XCEPTION Results (Grayscale):")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        xception_accuracy = test_accuracy
        xception_loss = test_loss
        
    except Exception as e2:
        print(f"❌ Error with grayscale too: {e2}")
        xception_accuracy = None
        xception_loss = None

# ============= Comparison =============
print("\n" + "="*70)
print("📈 ACCURACY COMPARISON")
print("="*70)

if custom_accuracy is not None and xception_accuracy is not None:
    improvement = custom_accuracy - xception_accuracy
    ratio = custom_accuracy / xception_accuracy if xception_accuracy > 0 else 0
    
    print(f"\nCustom CNN (100 Epochs, 48×48): {custom_accuracy*100:.2f}%")
    print(f"XCEPTION (64×64):               {xception_accuracy*100:.2f}%")
    print(f"\nImprovement: +{improvement*100:.2f} percentage points")
    print(f"Ratio: {ratio:.2f}× better")
    
    if custom_accuracy > xception_accuracy:
        print(f"\n✅ Custom model is SUPERIOR by {(ratio-1)*100:.1f}%")
    else:
        print(f"\n⚠️  XCEPTION is better by {(1/ratio-1)*100:.1f}%")
        
elif custom_accuracy is not None:
    print(f"\n✅ Custom CNN (100 Epochs): {custom_accuracy*100:.2f}%")
    print(f"❌ XCEPTION: Could not evaluate")
    
print("\n" + "="*70)

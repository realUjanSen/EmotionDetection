#!/usr/bin/env python3
"""
Continue Training Emotion Detection Model - Epochs 61-100
Loads the best model from 60 epochs and continues training
Plots full training history (1-100) at the end
"""

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    
    print("✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def main():
    print("\n🎭 CONTINUE EMOTION DETECTION TRAINING - EPOCHS 61-100")
    print("=" * 60)
    
    # Load the best model from 60 epochs
    model_path = 'models/emotion_model_60epochs.h5'
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"📁 Loading model from {model_path}...")
    try:
        model = load_model(model_path, compile=False)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Check data directories
    train_dir = os.path.join('data', 'fer2013', 'train')
    test_dir = os.path.join('data', 'fer2013', 'test')
    
    if not os.path.exists(train_dir):
        print(f"❌ Training data not found at {train_dir}")
        return
    if not os.path.exists(test_dir):
        print(f"❌ Test data not found at {test_dir}")
        return
    
    print(f"✅ Training data found")
    print(f"✅ Test data found\n")
    
    # Data generators
    print("📊 Setting up data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training'
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation'
    )
    
    # Test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical'
    )
    
    print(f"✅ Generators ready\n")
    
    # Callbacks
    print("⚙️ Setting up callbacks...")
    callbacks = [
        ModelCheckpoint(
            'models/emotion_model_100epochs.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Continue training (epochs 61-100, so initial_epoch=60, epochs=100)
    print("\n🚀 Starting training from epoch 61...")
    print("=" * 60)
    history = model.fit(
        train_generator,
        epochs=100,
        initial_epoch=60,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"✅ Test Loss: {test_loss:.4f}")
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    
    # Plot FULL training history (1-100 epochs)
    print("\n📈 Plotting FULL training history (Epochs 1-100)...")
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Model Loss (Epochs 1-100)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.title('Model Accuracy (Epochs 1-100)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_full_1-100epochs.png', dpi=150, bbox_inches='tight')
    print("✅ Full training history saved to training_history_full_1-100epochs.png")
    
    print("\n🎉 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"📁 Best model saved to: models/emotion_model_100epochs.h5")
    print(f"📊 Full training plot saved to: training_history_full_1-100epochs.png")
    print(f"✅ Final Test Accuracy: {test_accuracy:.2%}")

if __name__ == "__main__":
    main()

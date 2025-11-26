#!/usr/bin/env python3
"""
Simplified Emotion Detection Model Training - 50 Epochs with Early Stopping
Minimal imports to avoid compatibility issues
"""

import os
import sys

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    # Core imports
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    
    print("✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def create_model():
    """Create CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])
    return model

def main():
    print("\n🎭 EMOTION DETECTION MODEL TRAINING - 50 EPOCHS")
    print("=" * 60)
    
    # Check data directories
    train_dir = os.path.join('data', 'fer2013', 'train')
    test_dir = os.path.join('data', 'fer2013', 'test')
    
    if not os.path.exists(train_dir):
        print(f"❌ Training data not found at {train_dir}")
        return
    if not os.path.exists(test_dir):
        print(f"❌ Test data not found at {test_dir}")
        return
    
    print(f"✅ Training data found at {train_dir}")
    print(f"✅ Test data found at {test_dir}\n")
    
    # Data augmentation
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
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {validation_generator.samples}")
    print(f"✅ Test samples: {test_generator.samples}")
    print(f"✅ Classes: {list(train_generator.class_indices.keys())}\n")
    
    # Build model
    print("🏗️ Building CNN model...")
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("📋 Model Summary:")
    model.summary()
    
    # Callbacks
    print("\n⚙️ Setting up callbacks...")
    callbacks = [
        ModelCheckpoint(
            'models/emotion_model_50epochs.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\n🚀 Starting training...")
    print("=" * 60)
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"✅ Test Loss: {test_loss:.4f}")
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    print("\n📈 Plotting training history...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_50epochs.png', dpi=100, bbox_inches='tight')
    print("✅ Training history saved to training_history_50epochs.png")
    
    print("\n🎉 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"📁 Model saved to: models/emotion_model_50epochs.h5")
    print(f"📊 Training plot saved to: training_history_50epochs.png")
    print(f"✅ Final Test Accuracy: {test_accuracy:.2%}")

if __name__ == "__main__":
    main()

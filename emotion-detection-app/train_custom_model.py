#!/usr/bin/env python3
"""
Quick Emotion Detection Model Training Script
Train a CNN model on FER2013 dataset for 75 epochs
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

def create_emotion_model():
    """Create a CNN model for emotion detection"""
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Conv Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Conv Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Fourth Conv Block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(7, activation='softmax')  # 7 emotions
    ])
    
    return model

def setup_data_generators():
    """Setup data generators for training and validation"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        validation_split=0.2  # Use 20% of training data for validation
    )
    
    # Only rescaling for test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        'data/fer2013/train',
        target_size=(48, 48),
        batch_size=64,  # Increased batch size
        color_mode='grayscale',
        class_mode='categorical',
        subset='training'
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        'data/fer2013/train',
        target_size=(48, 48),
        batch_size=64,  # Increased batch size
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation'
    )
    
    # Test generator
    test_generator = test_datagen.flow_from_directory(
        'data/fer2013/test',
        target_size=(48, 48),
        batch_size=64,  # Increased batch size
        color_mode='grayscale',
        class_mode='categorical'
    )
    
    return train_generator, validation_generator, test_generator

def train_model():
    """Main training function"""
    print("🚀 Starting Emotion Detection Model Training...")
    print("📊 Setting up data generators...")
    
    # Setup data
    train_gen, val_gen, test_gen = setup_data_generators()
    
    print(f"📁 Found {train_gen.samples} training images")
    print(f"📁 Found {val_gen.samples} validation images")
    print(f"📁 Found {test_gen.samples} test images")
    print(f"🎭 Emotion classes: {list(train_gen.class_indices.keys())}")
    
    # Create model
    print("🏗️ Building CNN model...")
    model = create_emotion_model()
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    print("📋 Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'models/emotion_model_custom_75epochs.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train model
    print("🎯 Starting training for 75 epochs...")
    history = model.fit(
        train_gen,
        epochs=75,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("📊 Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
    print(f"📉 Test Loss: {test_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    model.save('models/emotion_model_final_75epochs.h5')
    print("💾 Model saved successfully!")
    
    return model, history

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_custom_75epochs.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Training history plot saved as 'training_history_custom_75epochs.png'")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("🎭 Custom Emotion Detection Model Training")
    print("=" * 50)
    
    # Check if GPU is available
    if tf.config.list_physical_devices('GPU'):
        print("🚀 GPU detected! Training will be faster.")
    else:
        print("⚠️ No GPU detected. Training will use CPU.")
    
    # Start training
    model, history = train_model()
    
    print("\n🎉 Training completed successfully!")
    print("📁 Model saved to: models/emotion_model_final_75epochs.h5")
    print("📊 Training plot saved to: training_history_custom_75epochs.png")

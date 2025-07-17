#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Evaluates all emotion detection models with confusion matrices and classification reports
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# Emotion classes
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def load_and_compile_model(model_path, img_size):
    """Load and compile a model"""
    print(f"Loading model: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

def create_test_generator(img_size, batch_size=64):
    """Create test data generator"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'data/fer2013/test',
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    return test_generator

def evaluate_model(model, test_generator, model_name):
    """Evaluate model and generate confusion matrix"""
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*50}")
    
    # Get predictions
    print("Making predictions...")
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_generator.classes
    
    # Calculate accuracy and loss
    loss, accuracy = model.evaluate(test_generator, verbose=0)
    
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Test Loss: {loss:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, 
                              target_names=emotion_classes))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_classes,
                yticklabels=emotion_classes)
    plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {accuracy:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save plot
    safe_name = model_name.replace('.', '_').replace('/', '_')
    plt.savefig(f'confusion_matrix_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, loss

def main():
    """Main evaluation function"""
    print("🎭 Comprehensive Model Evaluation")
    print("=" * 50)
    
    # Model configurations: (path, image_size, display_name)
    models_config = [
        ('models/fer2013_big_XCEPTION.54-0.66.hdf5', (64, 64), 'XCEPTION Model'),
        ('models/emotion_cnn_fixed.h5', (48, 48), 'CNN Fixed Model'),
        ('models/emotion_cnn_new.h5', (48, 48), 'CNN New Model')
    ]
    
    results = []
    
    for model_path, img_size, model_name in models_config:
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            continue
            
        # Load model
        model = load_and_compile_model(model_path, img_size)
        if model is None:
            continue
            
        # Create test generator
        test_generator = create_test_generator(img_size)
        
        # Evaluate model
        accuracy, loss = evaluate_model(model, test_generator, model_name)
        results.append((model_name, accuracy, loss))
        
        # Clear memory
        del model
        tf.keras.backend.clear_session()
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':<12} {'Loss':<10}")
    print("-" * 60)
    
    for model_name, accuracy, loss in results:
        print(f"{model_name:<20} {accuracy:.4f} ({accuracy*100:.1f}%) {loss:.4f}")
    
    if results:
        best_model = max(results, key=lambda x: x[1])
        print(f"\n🏆 Best Model: {best_model[0]} with {best_model[1]:.4f} accuracy")

if __name__ == "__main__":
    # Check if test data exists
    if not os.path.exists('data/fer2013/test'):
        print("❌ Test data not found at 'data/fer2013/test'")
        exit(1)
    
    main()
    print("\n✅ Evaluation complete! Check the confusion matrix images.")

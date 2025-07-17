import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import os
import numpy as np
from collections import Counter

# Set environment variables for maximum resource usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
os.environ['OMP_NUM_THREADS'] = '8'  # Adjust based on your CPU cores
os.environ['MKL_NUM_THREADS'] = '8'  # Adjust based on your CPU cores
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force GPU usage

# Configure TensorFlow threading
tf.config.threading.set_intra_op_parallelism_threads(8)  # Adjust based on your CPU cores
tf.config.threading.set_inter_op_parallelism_threads(8)  # Adjust based on your CPU cores

# Enable dynamic memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Debugging GPU usage
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
if not tf.config.list_physical_devices('GPU'):
    print("[DEBUG] No GPUs detected. Ensure TensorFlow is installed with GPU support and drivers are configured.")
else:
    print("[DEBUG] GPUs detected. TensorFlow will attempt to use them.")

# Debugging TensorFlow threading
print("[DEBUG] Intra-op parallelism threads:", tf.config.threading.get_intra_op_parallelism_threads())
print("[DEBUG] Inter-op parallelism threads:", tf.config.threading.get_inter_op_parallelism_threads())

# Debugging environment variables
print("[DEBUG] Environment variables:")
print("TF_CPP_MIN_LOG_LEVEL:", os.environ.get('TF_CPP_MIN_LOG_LEVEL'))
print("OMP_NUM_THREADS:", os.environ.get('OMP_NUM_THREADS'))
print("MKL_NUM_THREADS:", os.environ.get('MKL_NUM_THREADS'))
print("TF_ENABLE_ONEDNN_OPTS:", os.environ.get('TF_ENABLE_ONEDNN_OPTS'))
print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))

# Set paths
train_dir = os.path.join('data', 'fer2013', 'train')
test_dir = os.path.join('data', 'fer2013', 'test')

# Parameters
batch_size = 64  # Increased batch size for faster training
img_height = 48
img_width = 48

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale'  # Changed to grayscale for emotion detection
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale'  # Changed to grayscale for emotion detection
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# Calculate class weights to handle imbalance
def get_class_weights(dataset):
    class_counts = Counter()
    for images, labels in dataset:
        for label in labels:
            class_counts[label.numpy()] += 1
    
    total_samples = sum(class_counts.values())
    class_weights = {}
    for class_id, count in class_counts.items():
        class_weights[class_id] = total_samples / (num_classes * count)
    
    print("Class distribution:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {class_counts.get(i, 0)} samples, weight: {class_weights.get(i, 1.0):.2f}")
    
    return class_weights

class_weights = get_class_weights(train_ds)

# Data augmentation - more conservative for emotion detection
augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),  # Reduced rotation
    layers.RandomZoom(0.05),      # Reduced zoom
    layers.RandomContrast(0.1),   # Added contrast variation
])

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Build faster model for development
model = keras.Sequential([
    augmentation,
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
    
    # Simpler architecture for faster training
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    
    # Simpler dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile with better optimizer settings
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    loss='sparse_categorical_crossentropy',  # Changed loss function
    metrics=['accuracy']
)

model.summary()

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop training if no improvement for 5 epochs
    restore_best_weights=True
)

epochs = 35  # Increased number of epochs
print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs,
    class_weight=class_weights,  # Use class weights to handle imbalance
    verbose=1,
    callbacks=[early_stopping]  # Add early stopping
)

# Save the model
model.save(os.path.join('models', 'emotion_cnn_new.h5'))

# Evaluate the model
print("\nEvaluating model on test set...")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

print('\nTraining complete!')
print('Model saved to: models/emotion_cnn_new.h5')
print('\nClass mapping:')
for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

print("Training history plot saved as 'training_history.png'")

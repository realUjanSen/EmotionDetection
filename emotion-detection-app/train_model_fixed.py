import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Set environment variables for maximum CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
os.environ['OMP_NUM_THREADS'] = '8'  # Adjust based on your CPU cores
os.environ['MKL_NUM_THREADS'] = '8'  # Adjust based on your CPU cores
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU

# Configure TensorFlow threading
tf.config.threading.set_intra_op_parallelism_threads(8)  # Adjust based on your CPU cores
tf.config.threading.set_inter_op_parallelism_threads(8)  # Adjust based on your CPU cores

# Set paths
train_dir = os.path.join('data', 'fer2013', 'train')
test_dir = os.path.join('data', 'fer2013', 'test')

# Parameters
batch_size = 32
img_height = 48
img_width = 48
epochs = 20

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")
print(f"Class indices: {train_generator.class_indices}")

# Build a more robust model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.2),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile with a good optimizer
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Prefetch for performance
train_generator = train_generator.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_generator = validation_generator.prefetch(buffer_size=tf.data.AUTOTUNE)

# Train without class weights initially to see if the model can learn
print("Training model...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    verbose=1
)

# Evaluate
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Save the model
model.save('models/emotion_cnn_fixed.h5')
print("Model saved as: models/emotion_cnn_fixed.h5")

# Quick prediction test
print("\nTesting predictions...")
test_batch = next(test_generator)
predictions = model.predict(test_batch[0][:5])
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_batch[1][:5], axis=1)

class_names = list(train_generator.class_indices.keys())
print("Sample predictions:")
for i in range(5):
    print(f"True: {class_names[true_classes[i]]}, Predicted: {class_names[predicted_classes[i]]}")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Path to model and test data
test_data_dir = 'data/fer2013/test'
model_path = 'models/fer2013_big_XCEPTION.54-0.66.hdf5'

# Image parameters
img_size = (64, 64)  # XCEPTION model expects 64x64 images
batch_size = 64

# Load model
print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path, compile=False)

# Recompile model for evaluation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Prepare test data generator
print(f"Preparing test data from {test_data_dir}...")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

# Evaluate model
print("Evaluating model on test set...")
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

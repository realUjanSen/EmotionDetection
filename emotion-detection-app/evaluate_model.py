import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths and parameters
MODEL_PATH = os.path.join('models', 'fer2013_big_XCEPTION.54-0.66.hdf5')  # Updated model filename
TEST_DIR = os.path.join('data', 'fer2013', 'test')
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = (64, 64)  # CHANGED: XCEPTION model expects 64x64 images
BATCH_SIZE = 32

# Data generator for test set
val_datagen = ImageDataGenerator(rescale=1./255)
test_generator = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',  # FIXED: Changed back to grayscale to match training
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Load model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("🔧 Trying to load without compilation...")
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded without compilation!")
        # Recompile the model with current Keras version
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Model recompiled successfully!")
    except Exception as e2:
        print(f"❌ Failed to load model: {e2}")
        exit(1)

# Predict
Y_pred = model.predict(test_generator, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print confusion matrix as text
print('Confusion Matrix:')
print(cm)

# Print class indices mapping
print('Class indices mapping:', test_generator.class_indices)

# Print class distribution in test set
print('Test set class distribution:')
unique, counts = np.unique(y_true, return_counts=True)
for idx, count in zip(unique, counts):
    print(f"{EMOTIONS[idx]}: {count}")

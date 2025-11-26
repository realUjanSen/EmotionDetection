from keras.models import load_model
import cv2
import numpy as np

class EmotionPredictor:
    def __init__(self, model_path):
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Loading model without compilation due to: {e}")
            self.model = load_model(model_path, compile=False)
            # Recompile for current Keras version
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def preprocess_image(self, image):
        # Convert to grayscale if needed
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        
        # Custom trained model uses 48x48 (not XCEPTION's 64x64)
        resized_image = cv2.resize(image, (48, 48))
        normalized_image = resized_image / 255.0
        reshaped_image = np.reshape(normalized_image, (1, 48, 48, 1))
        return reshaped_image

    def predict_emotion(self, image):
        # Preprocess: Convert to grayscale, resize to 48x48, normalize
        processed_image = self.preprocess_image(image)
        # Run inference using loaded custom model
        predictions = self.model.predict(processed_image)
        # Return highest probability emotion label
        emotion_index = np.argmax(predictions)
        return self.emotion_labels[emotion_index], predictions[0][emotion_index]
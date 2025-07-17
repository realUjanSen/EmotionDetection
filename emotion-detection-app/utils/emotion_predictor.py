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
        
        resized_image = cv2.resize(image, (64, 64))  # CHANGED: XCEPTION expects 64x64
        normalized_image = resized_image / 255.0
        reshaped_image = np.reshape(normalized_image, (1, 64, 64, 1))  # CHANGED: 64x64x1
        return reshaped_image

    def predict_emotion(self, image):
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image)
        emotion_index = np.argmax(predictions)
        return self.emotion_labels[emotion_index], predictions[0][emotion_index]
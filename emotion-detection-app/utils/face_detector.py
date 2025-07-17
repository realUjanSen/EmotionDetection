import cv2
import numpy as np

class FaceDetector:
    def __init__(self, model_type='haar'):
        if model_type == 'haar':
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        elif model_type == 'dnn':
            self.net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
        else:
            raise ValueError("Invalid model type. Choose 'haar' or 'dnn'.")

    def detect_faces(self, image):
        if hasattr(self, 'face_cascade'):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            return [(x, y, x+w, y+h) for (x, y, w, h) in faces]
        elif hasattr(self, 'net'):
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    faces.append((startX, startY, endX, endY))
            return faces
        return []

    def draw_faces(self, image, faces):
        for (startX, startY, endX, endY) in faces:
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        return image
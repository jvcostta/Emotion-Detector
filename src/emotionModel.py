import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess_face

class EmotionModel:
    def __init__(self, model_path, labels):
        self.model = load_model(model_path)
        self.labels = labels

    def predict_emotion(self, face):
        face_input = preprocess_face(face)
        prediction = self.model.predict(face_input)
        return self.labels[np.argmax(prediction)]

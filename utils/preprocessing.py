import cv2
import numpy as np

def preprocess_face(face):
    face_resized = cv2.resize(face, (48, 48))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb / 255.0
    face_input = np.expand_dims(face_normalized, axis=0)
    return face_input

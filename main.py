from src.emotionModel import EmotionModel
from src.videoCapture import start_video_stream

labels = ["Raiva", "Feliz", "Triste"]
model_path = "model/emotion_model.h5"

emotion_model = EmotionModel(model_path, labels)
start_video_stream(emotion_model, labels)

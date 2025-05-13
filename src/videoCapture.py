import cv2
from src.faceDetection import detect_faces

def start_video_stream(model, emotion_labels):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        faces = detect_faces(frame)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emotion = model.predict_emotion(face)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

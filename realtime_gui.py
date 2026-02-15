import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load emotion model
model = load_model("models/emotion_model.keras")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# -------- MediaPipe NEW API --------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# Webcam
cap = cv2.VideoCapture(0)
timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 1

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:

            h, w, _ = frame.shape

            x_coords = [int(lm.x * w) for lm in face_landmarks]
            y_coords = [int(lm.y * h) for lm in face_landmarks]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

            face = frame[y_min:y_max, x_min:x_max]

            if face.size != 0:
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_gray = cv2.resize(face_gray, (48, 48))
                face_gray = face_gray / 255.0
                face_gray = np.reshape(face_gray, (1, 48, 48, 1))

                prediction = model.predict(face_gray, verbose=0)
                emotion = emotion_labels[np.argmax(prediction)]

                cv2.putText(frame, emotion,
                            (x_min, y_min-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0,255,0), 2)

            # Draw landmarks
            for lm in face_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

    cv2.imshow("Emotion Detection - Advanced Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

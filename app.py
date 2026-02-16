import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/emotion_model.keras")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

st.title("Real-Time Emotion Detection")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))

    prediction = model.predict(face)
    emotion = emotion_labels[np.argmax(prediction)]

    cv2.putText(frame, emotion, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)

    FRAME_WINDOW.image(frame, channels="BGR")

camera.release()

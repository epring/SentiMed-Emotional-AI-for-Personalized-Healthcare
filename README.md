# SentiMed-Emotional-AI-for-Personalized-Healthcare
SentiMed uses emotion recognition and ML to analyze patients' emotional states during telemedicine sessions, providing doctors with valuable insights to improve patient care and mental well-being.
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from deepface import DeepFace
import streamlit as st

# Load pre-trained emotion recognition model
# This model should be trained on facial expression datasets like FER2013
emotion_model_path = 'models/emotion_model.h5'
emotion_model = load_model(emotion_model_path)

# Streamlit UI for telemedicine sessions
def main():
    st.title("SentiMed: Emotional AI for Personalized Healthcare")

    # Start webcam
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        success, frame = camera.read()
        if not success:
            break

        # Convert frame to format suitable for model prediction
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = DeepFace.detectFace(frame, enforce_detection=False)

        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            # Predict emotion
            prediction = emotion_model.predict(face)
            max_index = np.argmax(prediction)
            emotion = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'][max_index]

            # Display emotion on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, emotion, (x, y-10), font, 0.9, (0,255,0), 2)

        # Display the frame with detected emotion
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    else:
        st.write('Stopped')

if __name__ == '__main__':
    main()

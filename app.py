import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model("modelFEC.h5", compile=False)
classes = ['angry','disgust','fear','happy','neutral','sad','surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = model.predict(roi)[0]
            idx = np.argmax(preds)
            label = classes[idx]
            confidence = preds[idx]
            text = f"{label}: {confidence*100:.0f}%"
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 200, 0), 2)
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        return img

st.title("🧠 Detector de Emociones")
st.markdown("Activa tu cámara y detecta emociones en tiempo real")
webrtc_streamer(key="emotion", video_transformer_factory=EmotionDetector)

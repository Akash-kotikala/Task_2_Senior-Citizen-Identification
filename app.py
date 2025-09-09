import cv2
import numpy as np
import pandas as pd
import datetime
import os
import tempfile
import streamlit as st
from tensorflow.keras.models import load_model

# -------------------------
# Load Pretrained Model
from tensorflow.keras.models import load_model, save_model

# Load with old compatibility
model1 = load_model("age_gender_model.h5", compile=False)

# Save in new format
model1.save("age_gender_model.keras", save_format="keras")
from keras.models import load_model
model = load_model("age_gender_model.keras", compile=False)

# -------------------------
# Prediction Function
# -------------------------
def predict_age_gender(face_img, model):
    img = cv2.resize(face_img, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    age_pred, gender_pred = model.predict(img, verbose=0)

    age = int(age_pred[0][0])   # regression output for age
    gender = "Female" if np.argmax(gender_pred[0]) == 1 else "Male"
    return age, gender

# -------------------------
# CSV Logger
# -------------------------
def log_visit(age, gender, filename="visitors.csv"):
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[age, gender, time_now]], 
                      columns=["Age", "Gender", "Time"])
    
    if os.path.exists(filename):
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        df.to_csv(filename, mode="w", header=True, index=False)

# -------------------------
# Detection Function
# -------------------------
def run_detection(source):
    cap = cv2.VideoCapture(source)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    stframe = st.empty()  # streamlit video placeholder

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            try:
                age, gender = predict_age_gender(face, model)

                if age > 60:
                    color = (0, 0, 255)
                    label = f"{gender}, {age} yrs (Senior)"
                else:
                    color = (0, 255, 0)
                    label = f"{gender}, {age} yrs"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                log_visit(age, gender)

            except Exception as e:
                print("Prediction error:", e)
                continue

        stframe.image(frame, channels="BGR")

    cap.release()

# -------------------------
# Streamlit GUI
# -------------------------
st.title("🎥 Senior Citizen Identification")
st.write("Detect Age, Gender, and mark Senior Citizens (>60 yrs)")

mode = st.radio("Choose Input Source:", ["Webcam", "Upload Video"])

if mode == "Webcam":
    if st.button("Start Webcam"):
        run_detection(0)

elif mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        if st.button("Start Video Processing"):
            run_detection(tfile.name)

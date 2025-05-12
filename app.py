from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('C:/Users/user/OneDrive/Desktop/ML/computer_vision/selfie_detection/model.h5')

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Ensure this file is in same dir

# Create a folder to save the images if it doesn't exist
SAVE_FOLDER = 'saved_images'
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to capture the frame from the webcam and predict smile
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (28, 28))
            face_normalized = face_resized.astype("float") / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 28, 28, 1))

            prediction = model.predict(face_reshaped)[0]
            label = "Smiling" if prediction[1] > 0.5 else "Not Smiling"

            color = (0, 255, 0) if label == "Smiling" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if label == "Smiling":
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                save_path = os.path.join(SAVE_FOLDER, f"{timestamp}_smiling.jpg")
                cv2.imwrite(save_path, frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Must be inside a /templates folder

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

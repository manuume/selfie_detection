# 😊 Smile Detection Selfie Web App

This is a fun AI-powered web application that **automatically takes a selfie when you smile**! It uses a **CNN model (LeNet)** trained to detect smiles in real time via webcam, and then **saves the smiling image** to a folder.

## 🔍 Features

* Uses your **webcam** to stream live video.
* Detects **faces** using Haar Cascade.
* Detects **smiles** using a **custom CNN model** trained on the SMILE dataset.
* When a smile is detected:

  * It shows a green rectangle around your face.
  * It **saves the image** to a `saved_images` folder with a timestamp.
* Real-time video is streamed in the browser.

## 💠 Tech Stack

* **Python**
* **Flask** (for the web interface)
* **OpenCV** (for face detection and image processing)
* **Keras + TensorFlow** (for smile classification using CNN)
* **HTML** (basic frontend)

## 📁 Project Structure

```
selfie_detection/
│
├── app.py                  # Main Flask app
├── model.h5                # Trained CNN model (LeNet)
├── saved_images/           # Captured smiling selfies
├── static/                 # (Optional for styling/images)
├── templates/
│   └── index.html          # Web interface
├── smiles/                 # Dataset folder (positives/negatives)
├── haarcascade_frontalface_default.xml  # Face detection model
├── .gitignore              # Excludes large folders like `data/`
└── README.md               # Project documentation
```

## 🚀 How to Run

1. Make sure you have Python 3 installed.
2. Install dependencies:

```bash
pip install flask opencv-python keras tensorflow
```

3. Put the `haarcascade_frontalface_default.xml` file in the root directory (same folder as `app.py`).
4. Run the app:

```bash
python app.py
```

5. Open your browser and go to: `http://127.0.0.1:5000/`

6. **Smile!** 😊 — it will take your selfie and save it to the `saved_images/` folder.

## 📌 Notes

* The app only saves photos **when a face is detected and the person is smiling**.
* Make sure your webcam is connected and allowed by the browser.

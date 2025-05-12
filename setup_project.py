import os

# Define the root folder path where your project is located
root_dir = r"C:\Users\user\OneDrive\Desktop\ML\computer vision\selfie_detection"

# List of required folders to create
folders = [  # Folder to hold the dataset
    "lenet",  # Folder to store the model's code
    "lenet/nn",  # Subfolder for neural network models
    "lenet/nn/conv",  # Subfolder for convolutional models
]

# Create the folders
for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created folder: {folder_path}")

# Define files to create with their content
files = {
    "train_model.py": '''import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from imutils import paths
import imutils

# Initialize dataset path
dataset_path = "smiles/dataset"

# Initialize the list of data and labels
data = []
labels = []

# Load and preprocess images
for imagePath in sorted(list(paths.list_images(dataset_path))):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = np.array(image, dtype='float32') / 255.0
    data.append(image)

    label = imagePath.split(os.path.sep)[-3]
    label = 'smiling' if label == 'positives' else 'not_smiling'
    labels.append(label)

# Convert labels to one-hot encoding
le = LabelEncoder()
labels = np_utils.to_categorical(le.fit_transform(labels), 2)

# Split dataset into train and test
trainX, testX, trainY, testY = train_test_split(np.array(data), labels, test_size=0.2, stratify=labels, random_state=42)

# Initialize the model
model = Sequential()
model.add(Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=10)

# Save the model
model.save('smile_detection_model.h5')
print("Model saved as 'smile_detection_model.h5'")
''',
    "app.py": '''from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model = load_model('smile_detection_model.h5')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Read and process image
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0) / 255.0

    # Predict
    prediction = model.predict(img)
    label = 'smiling' if prediction.argmax() == 1 else 'not_smiling'

    return label

if __name__ == '__main__':
    app.run(debug=True)
''',
    "index.html": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smile Detection</title>
</head>
<body>
    <h1>Smile Detection</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <button type="submit">Upload Image</button>
    </form>
</body>
</html>
''',
    "lenet/nn/conv/__init__.py": '''from .lenet import LeNet
''',
    "lenet/nn/conv/lenet.py": '''from keras import layers
from keras.models import Sequential

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        model.add(layers.Conv2D(20, (5, 5), activation='relu', input_shape=(height, width, depth)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dense(classes, activation='softmax'))
        return model
''',
}

# Create the files and write the content
for file_path, content in files.items():
    full_path = os.path.join(root_dir, file_path)
    with open(full_path, 'w') as file:
        file.write(content)
    print(f"Created file: {full_path}")

print("Folders and files have been created successfully.")

# import necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from lenet.nn.conv import LeNet
from imutils import paths
import imutils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, 
    help='path to the input dataset of faces')
ap.add_argument('-m', '--model', required=True, 
    help='path to output model')
args = vars(ap.parse_args())

# initialize the list of data and labels
data = []
labels = []

# loop over the input images
for imagePath in sorted(list(paths.list_images(args['dataset']))):
    # load the image, pre-process it, and store in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28) # 28 x 28 x 1
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the labels list
    label = imagePath.split(os.path.sep)[-3] # C:\Users\Balaji\Documents\Smile-Detector\SMILEs\positives\positives7\3.jpg
    label = 'smiling' if label == 'positives' else 'not_smiling'
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype='float') / 255.0 # 0 to 255
labels = np.array(labels)

# convert the labels from integers to vectors using LabelEncoder and to_categorical
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), 2)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = dict()

for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# initialize the model
print('[INFO] compiling model...')
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the network
print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight=classWeight, batch_size=64, epochs=15, verbose=1)

# evaluate the network
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# save the model to disk
print('[INFO] serializing network')
model.save(args['model'])

# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 15), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 15), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 15), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, 15), H.history['val_accuracy'], label='val_accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()

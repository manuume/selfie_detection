from keras import layers
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

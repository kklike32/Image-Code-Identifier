"""
    This file defines the model to be used for training, transferring learning, and then running the model 
    on student handwriting results.
"""
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np


print(" ::: STARTED MODEL TRAINING ::: ")


"""
    Data Engineering :: this part of the code will load MNIST data, engineer it to optimize training, 
                        then load our new data and engineer it to fit the specifications of the MNIST 
                        data as best as we can.
"""
def load_mnist():
    # load mnist for transfer learning
    print("loading data for transfer learning...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # process data for optimal character recognition
    print("processing data for optimal results...")
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, x_test, y_train, y_test


def load_convex_data():
    # load student handwriting after it's processed
    print("loading data for retraining...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # process data for optimal character recognition
    print("processing data for optimal results...")
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, x_test, y_train, y_test

"""
    Model Deployment :: this part of the code will train the model on the MNIST data, save the resulting 
                        weights for easy initialization in the future, then transfer that learning to 
                        the new data for retraining. Throughout the process, weights will be sequentially 
                        saved in order to preserve progress and record evolution (for future optimization).
"""
def model_architecture():
    # define model architecture
    print("defining model architecture...")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # compile model
    print("compiling model...")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # return model
    return model


def train_mnist(model):
    # train model on MNIST
    print("training on MNIST data...")
    model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test))

    # save weights
    save_weights(model)


def save_weights(model):
    # save weights
    print("saving MNIST weights...")
    weights = model.get_weights()
    with open("mnist_weights.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for weight in weights:
            writer.writerow(weight.flatten())
        

def load_weights(filepath):
    # load from file
    print("loading weights for transfer learning...")
    with open(filepath, 'r') as csvfile:
    reader = csv.reader(csvfile)
    weights = []
    for row in reader:
        weights.append(row.astype(float))
    
    # return weights
    return weights

    
"""
    Transfer the learning
"""
def train_convex_data(model):
    # load weights for transfer learning
    print("transferring learning & retraining...")
    transfer_weights = load_weights("mnist_weights.csv")
    model.set_weights(weights)

    # train on new data
    print("training on pre-processed student handwriting data...")
    model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test))


"""
    Model Testing :: ChatGPT produced model testing example
"""
example_index = 0
example_image = x_test[example_index]
example_label = y_test[example_index]

# Reshape the image to match the input shape of the model
example_image = example_image.reshape(1, 28, 28, 1)

# Make a prediction
prediction = model.predict(example_image)

# Get the predicted label (the index with the highest probability)
predicted_label = tf.argmax(prediction, axis=1)

print("Example:")
print("True Label:", tf.argmax(example_label))
print("Predicted Label:", predicted_label.numpy()[0])

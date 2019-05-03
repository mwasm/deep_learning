from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical


""" Step 1: Load the data-set"""
# Loading the MNIST data-set.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

""" Step 2: Build a neural nets containing layers."""
# The network architecture. Build a neural nets model containing two layers.
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

""" Step 3: Set the compilation parameters. """
# The compilation step.
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

""" Step 4: Convert the image data from uint8 to float32: range [0, 1]. """
# Preparing the image data.
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

""" Step 5: Convert the labels to categorical data. """
# Convert the labels to categorical data.
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

""" Step 6: Train & test the model"""
# Train the model
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Test the model on test data.
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('Test_loss: ', test_loss)
print('Test_acc: ', test_acc)
from keras.datasets import imdb
from functions import vectorize_sequences
import numpy as np
from keras import models, layers
import matplotlib.pyplot as plt

# Import the data.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Prepare the data.
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Vectorize the labels.
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Model implementation.
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compiling the model.
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Setting aside a validation set.
x_val = x_train[: 10000]
partial_x_train = x_train[10000:]
y_val = y_train[: 10000]
partial_y_train = y_train[10000:]

# Training your model.
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history
print(history_dict.keys())

# Plot the data.
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plotting the training and validation accuracy
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
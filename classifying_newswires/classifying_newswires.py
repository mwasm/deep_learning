from keras.datasets import reuters
from functions import vectorize_sequences
from keras.utils.np_utils import to_categorical
from keras import models, layers
import matplotlib.pyplot as plt

# Load the data.
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# Prepare the data.
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Vectorize the labels.
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# Building the model.
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# Compiling the model.
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Setting aside a validation set.
x_val = x_train[:5000]
partial_x_train = x_train[5000:]

y_val = one_hot_train_labels[:5000]
partial_y_train = one_hot_train_labels[5000:]

# Fit the model.
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

# Plotting the training and validation loss.
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plotting the training and validation accuracy.
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

predictions = model.predict(x_test)
print(predictions[0].shape)
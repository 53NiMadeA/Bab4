# Exercise 4.7 - 64 Filters
# MNIST CNN

import time
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape
X_train = X_train.reshape(
    (X_train.shape[0], 28, 28, 1)
).astype('float32')

X_test = X_test.reshape(
    (X_test.shape[0], 28, 28, 1)
).astype('float32')

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# One hot encode
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]

# Create model
model = Sequential()

model.add(Conv2D(
    64,
    (5, 5),
    input_shape=(28, 28, 1),
    activation='relu'
))

model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Training
start_time = time.time()

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200
)

training_time = time.time() - start_time

# Evaluation
scores = model.evaluate(X_test, y_test, verbose=0)

print("\n=== HASIL 64 FILTER, KERNEL 5x5 ===")
print("Training Time: %.2f detik" % training_time)
print("Accuracy: %.2f%%" % (scores[1] * 100))
print("CNN Error: %.2f%%" % ((1 - scores[1]) * 100))
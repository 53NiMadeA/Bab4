# Exercise 4.4 - LeNet dengan jumlah filter berbeda

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D

model = Sequential()

# Convolution Layer 1
model.add(Conv2D(
    filters=8,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(32, 32, 1)
))

# Pooling Layer 1
model.add(AveragePooling2D(pool_size=(2, 2)))

# Convolution Layer 2
model.add(Conv2D(
    filters=20,
    kernel_size=(3, 3),
    activation='relu'
))

# Pooling Layer 2
model.add(AveragePooling2D(pool_size=(2, 2)))

# Flatten
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))

# Output Layer
model.add(Dense(units=10, activation='softmax'))

# Menampilkan struktur model
model.summary()
# Exercise 4.5 - AlexNet dengan jumlah Dense unit berbeda

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Membuat model AlexNet
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(
    filters=96,
    input_shape=(224, 224, 3),
    kernel_size=(11, 11),
    activation='relu',
    strides=(4, 4),
    padding='valid'
))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='valid'
))

# 2nd Convolutional Layer
model.add(Conv2D(
    filters=256,
    kernel_size=(11, 11),
    activation='relu',
    strides=(1, 1),
    padding='valid'
))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='valid'
))

# 3rd Convolutional Layer
model.add(Conv2D(
    filters=384,
    kernel_size=(3, 3),
    activation='relu',
    strides=(1, 1),
    padding='valid'
))

# 4th Convolutional Layer
model.add(Conv2D(
    filters=384,
    kernel_size=(3, 3),
    activation='relu',
    strides=(1, 1),
    padding='valid'
))

# 5th Convolutional Layer
model.add(Conv2D(
    filters=256,
    kernel_size=(3, 3),
    activation='relu',
    strides=(1, 1),
    padding='valid'
))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='valid'
))

# Fully Connected Layer
model.add(Flatten())

# Diubah dari 4096 menjadi 2048
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))

# Diubah dari 4096 menjadi 2048
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))

# Diubah dari 1000 menjadi 512
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))

# Output Layer tetap 17
model.add(Dense(17, activation='softmax'))

# Menampilkan struktur model
model.summary()

# Compile model
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy']
)
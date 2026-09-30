# Exercise 4.7
# Menambahkan Conv2D, MaxPooling2D, dan Dropout

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Create model
model = Sequential()

# Layer awal
model.add(Conv2D(
    32,
    (5, 5),
    input_shape=(28, 28, 1),
    activation='relu'
))

model.add(MaxPooling2D())
model.add(Dropout(0.2))

# Layer tambahan
model.add(Conv2D(
    64,
    (3, 3),
    activation='relu'
))

model.add(MaxPooling2D())
model.add(Dropout(0.2))

# Fully Connected Layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(model.summary())
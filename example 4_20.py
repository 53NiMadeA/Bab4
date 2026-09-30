import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# Mengubah data menjadi matriks dataset
def convertToMatrix(data, step):
    X, Y = [], []

    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d,])
        Y.append(data[d,])

    return np.array(X), np.array(Y)


# Membaca dataset Airline Passengers
df = pd.read_csv(
    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv',
    usecols=[1],
    engine='python'
)

# Menampilkan 5 data pertama
print(df.head())

# Menampilkan grafik data
plt.plot(df)
plt.title('Data Jumlah Penumpang Pesawat')
plt.xlabel('Bulan')
plt.ylabel('Jumlah Penumpang')
plt.show()


# Menentukan jumlah data sebelumnya yang digunakan untuk prediksi
step = 4

N = df.shape[0]

# 80% data untuk training
Tp = int(df.shape[0] * 0.8)

values = df.values

train = values[0:Tp, :]
test = values[Tp:N, :]

# Menambahkan step data terakhir
test = np.append(test, np.repeat(test[-1, ], step))
train = np.append(train, np.repeat(train[-1, ], step))

# Membuat data input dan target
trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test, step)

# Mengubah bentuk data agar sesuai dengan input SimpleRNN
trainX = np.reshape(
    trainX,
    (trainX.shape[0], 1, trainX.shape[1])
)

testX = np.reshape(
    testX,
    (testX.shape[0], 1, testX.shape[1])
)


# Membuat model SimpleRNN
model = Sequential()

model.add(
    SimpleRNN(
        units=32,
        input_shape=(1, step),
        activation="relu"
    )
)

model.add(Dense(8, activation="relu"))
model.add(Dense(1))

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer='rmsprop'
)

# Menampilkan struktur model
model.summary()


# Melatih model
model.fit(
    trainX,
    trainY,
    epochs=10,
    batch_size=16,
    verbose=2
)


# Prediksi data training dan testing
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Menggabungkan hasil prediksi
predicted = np.concatenate(
    (trainPredict, testPredict),
    axis=0
)

# Menghitung nilai loss training
trainScore = model.evaluate(
    trainX,
    trainY,
    verbose=0
)

print("Training Loss:", trainScore)


# Menampilkan hasil prediksi
index = df.index.values

plt.plot(df, label='Data')
plt.plot(predicted, label='Prediction')

# Garis batas antara data training dan testing
plt.axvline(
    df.index[Tp],
    c="r"
)

plt.title('Prediksi Jumlah Penumpang dengan SimpleRNN')
plt.legend()
plt.show()
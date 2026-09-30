import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Memuat dataset
whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target

# Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data,
    test_size=0.3,
    random_state=7
)

# Membuat model dengan 5 neuron pada setiap Dense layer
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=[X_train.shape[1]]))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mse']
)

# Mengukur waktu training
start_time = time.time()

model.fit(
    X_train,
    y_train,
    batch_size=50,
    validation_split=0.2,
    epochs=100,
    verbose=1
)

end_time = time.time()

training_time = end_time - start_time

# Evaluasi model
results = model.evaluate(X_test, y_test)

print("\n=== HASIL EXERCISE 4.3 ===")
print("Training time:", training_time, "detik")
print("Loss:", results[0])
print("MSE:", results[1])
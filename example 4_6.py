# Example 4.6 - InceptionV3

from tensorflow.keras.applications import InceptionV3

# Inisialisasi model InceptionV3
model = InceptionV3(weights='imagenet')

# Menampilkan struktur model
model.summary()
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims

# Load model VGG19
model = VGG19(weights='imagenet')

# Pilih hidden layer yang akan divisualisasikan
n = 1

# Buat model baru yang menghasilkan output dari hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[n].output)

model.summary()

# Load gambar dengan ukuran yang sesuai VGG19
img = load_img('Elephant.jpg', target_size=(224, 224))

# Ubah gambar menjadi array
img = img_to_array(img)

# Tambahkan dimensi batch
img = expand_dims(img, axis=0)

# Preprocessing gambar untuk VGG19
img = preprocess_input(img)

# Dapatkan feature maps dari hidden layer
feature_maps = model.predict(img)

print("Feature maps:", feature_maps.shape)

# Tampilkan semua feature maps
plot_feature_maps(feature_maps)
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims

# Fungsi untuk menampilkan feature maps
def plot_feature_maps(feature_maps):
    col = 8
    row = int(feature_maps.shape[3] / col)
    ix = 1

    plt.figure(figsize=(20, 20))

    for _ in range(row):
        for _ in range(col):
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1

    plt.show()


# Load model VGG19
model = VGG19(weights='imagenet')

# Pilih convolutional layer
n = 2

# Model menghasilkan output dari layer yang dipilih
model = Model(inputs=model.inputs, outputs=model.layers[n].output)

print("Layer yang digunakan:", model.layers[-1].name)
model.summary()

# Load gambar
img = load_img('Elephant.jpg', target_size=(224, 224))

# Convert gambar menjadi array
img = img_to_array(img)

# Tambahkan dimensi batch
img = expand_dims(img, axis=0)

# Preprocessing gambar
img = preprocess_input(img)

# Mendapatkan feature maps
feature_maps = model.predict(img)

print("Feature maps:", feature_maps.shape)

# Menampilkan feature maps
plot_feature_maps(feature_maps)
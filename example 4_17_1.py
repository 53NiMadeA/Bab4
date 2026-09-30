# Example 4.17 CNN Visualize Filters

from keras.applications.vgg19 import VGG19

# Load VGG19 model
model = VGG19(weights='imagenet')

# Menampilkan struktur model
model.summary()
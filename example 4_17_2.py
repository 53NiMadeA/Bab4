from keras.applications.vgg19 import VGG19

model = VGG19(weights='imagenet')
model.summary()

n = 0
for layer in model.layers:
    print(n, layer.name)
    n += 1
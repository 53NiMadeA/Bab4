# Example 4.16 AutoencoderKeras.py

# Load dataset MNIST
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

# Normalisasi nilai pixel dari 0-255 menjadi 0-1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print("Shape data training :", x_train.shape)
print("Shape data testing  :", x_test.shape)


# ============================================================
# Build Autoencoder
# ============================================================

from keras.layers import Dense, Flatten, Reshape, Input
from keras.models import Sequential, Model


def build_autoencoder(img_shape, code_size):

    # Encoder
    encoder = Sequential()
    encoder.add(Input(shape=img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    # Decoder
    decoder = Sequential()
    decoder.add(Input(shape=(code_size,)))
    decoder.add(Dense(np.prod(img_shape)))
    decoder.add(Reshape(img_shape))

    return encoder, decoder


# Ukuran gambar
IMG_SHAPE = x_train[0].shape

# Ukuran representasi kode
encoder, decoder = build_autoencoder(
    IMG_SHAPE,
    32
)


# Menghubungkan Encoder dan Decoder
inp = Input(shape=IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(
    inp,
    reconstruction
)


# Compile Autoencoder
autoencoder.compile(
    optimizer='adamax',
    loss='mse'
)

print(autoencoder.summary())


# ============================================================
# Train Autoencoder
# ============================================================

history = autoencoder.fit(
    x=x_train,
    y=x_train,
    epochs=20,
    validation_data=(x_test, x_test)
)


# ============================================================
# Plot Training Results
# ============================================================

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(
    ['Train', 'Test'],
    loc='upper left'
)

plt.show()


# ============================================================
# Visualisasi Original, Code, dan Reconstructed Image
# ============================================================

def show_image(x):
    plt.imshow(
        np.clip(x, 0, 1),
        cmap='gray'
    )


def visualize(img, encoder, decoder):

    # Encode gambar
    code = encoder.predict(
        img[None],
        verbose=0
    )[0]

    # Decode kembali
    reco = decoder.predict(
        code[None],
        verbose=0
    )[0]

    # Original
    plt.subplot(1, 3, 1)
    plt.title("Original")
    show_image(img)

    # Code
    plt.subplot(1, 3, 2)
    plt.title("Code")
    plt.imshow(
        code.reshape([code.shape[-1] // 2, -1]),
        cmap='gray'
    )

    # Reconstructed
    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    show_image(reco)

    plt.show()


# Menampilkan 5 gambar
for i in range(5):
    img = x_test[i]
    visualize(
        img,
        encoder,
        decoder
    )
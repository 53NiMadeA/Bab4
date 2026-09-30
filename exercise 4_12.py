import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3,
    densenet
)

# =========================
# PILIH MODEL
# =========================

# VGG16
# model = vgg16.VGG16(weights='imagenet')

# ResNet50
# model = resnet50.ResNet50(weights='imagenet')

# MobileNet
# model = mobilenet.MobileNet(weights='imagenet')

# InceptionV3
# model = inception_v3.InceptionV3(weights='imagenet')

# DenseNet121
model = densenet.DenseNet121(weights='imagenet')

print(model.summary())

camera = cv2.VideoCapture(0)

# DenseNet121 menggunakan input 224 x 224
image_size = 224

while camera.isOpened():

    ok, cam_frame = camera.read()

    if not ok:
        print("Tidak dapat membaca kamera.")
        break

    # Resize frame
    frame = cv2.resize(
        cam_frame,
        (image_size, image_size)
    )

    # Mengubah gambar menjadi array
    numpy_image = img_to_array(frame)

    # Menambahkan dimensi batch
    image_batch = np.expand_dims(
        numpy_image,
        axis=0
    )

    # Preprocessing DenseNet
    processed_image = densenet.preprocess_input(
        image_batch.copy()
    )

    # Prediksi
    predictions = model.predict(
        processed_image,
        verbose=0
    )

    # Mengambil hasil prediksi
    label = decode_predictions(
        predictions,
        top=1
    )

    class_name = label[0][0][1]
    confidence = label[0][0][2]

    # Menampilkan hasil pada webcam
    cv2.putText(
        cam_frame,
        "{}, {:.1f}".format(
            class_name,
            confidence
        ),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow(
        'DenseNet Webcam Classification',
        cam_frame
    )

    # Tekan ESC untuk keluar
    key = cv2.waitKey(30)

    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
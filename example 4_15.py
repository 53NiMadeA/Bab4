import cv2
import numpy as np

from keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers


# =========================
# PILIH MODEL
# =========================

# VGG16
clf, preprocess_input = Classifiers.get('vgg16')

# ResNet50
# clf, preprocess_input = Classifiers.get('resnet50')

# MobileNetV2
# clf, preprocess_input = Classifiers.get('mobilenetv2')

# DenseNet201
# clf, preprocess_input = Classifiers.get('densenet201')

# InceptionV3
# clf, preprocess_input = Classifiers.get('inceptionv3')


# Ukuran input gambar
sz = 224

# Untuk InceptionV3 gunakan:
# sz = 299


# =========================
# MEMBUAT MODEL
# =========================

model = clf(
    input_shape=(sz, sz, 3),
    weights='imagenet',
    classes=1000
)

model.summary()


# =========================
# MEMBUKA WEBCAM
# =========================

camera = cv2.VideoCapture(0)

while True:

    ret, cam_frame = camera.read()

    if not ret:
        print("Tidak dapat membaca kamera.")
        break

    # Resize gambar
    frame = cv2.resize(
        cam_frame,
        (sz, sz)
    )

    # Mengubah gambar menjadi array
    image = np.asarray(frame)

    # Menambahkan dimensi batch
    image = np.expand_dims(image, 0)

    # Preprocessing
    image = preprocess_input(image)

    # Prediksi
    preds = model.predict(
        image,
        verbose=0
    )

    # Mengambil hasil prediksi
    label = decode_predictions(
        preds,
        top=1
    )

    class_name = label[0][0][1]
    confidence = label[0][0][2]

    # Menampilkan hasil
    cv2.putText(
        cam_frame,
        "{}, {:.1f}".format(
            class_name,
            confidence
        ),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Classification",
        cam_frame
    )

    # Tekan ESC untuk keluar
    key = cv2.waitKey(30)

    if key == 27:
        break


camera.release()
cv2.destroyAllWindows()
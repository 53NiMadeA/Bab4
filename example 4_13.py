# Example 4.13
# Real-time Object Classification menggunakan Webcam dan VGG16

import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import vgg16

# Ukuran input VGG16
image_size = 224

# Load model VGG16
model = vgg16.VGG16(weights='imagenet')

print(model.summary())

# Membuka webcam
camera = cv2.VideoCapture(0)

while camera.isOpened():

    # Membaca frame dari webcam
    ok, cam_frame = camera.read()

    if not ok:
        print("Tidak dapat membaca kamera.")
        break

    # Resize frame menjadi 224x224
    frame = cv2.resize(cam_frame, (image_size, image_size))

    # Convert image menjadi array
    numpy_image = img_to_array(frame)

    # Menambahkan dimensi batch
    image_batch = np.expand_dims(numpy_image, axis=0)

    # Preprocessing untuk VGG16
    processed_image = vgg16.preprocess_input(image_batch.copy())

    # Prediksi
    predictions = model.predict(processed_image, verbose=0)

    # Decode hasil prediksi
    label = decode_predictions(predictions, top=1)

    # Mengambil label dan confidence
    class_name = label[0][0][1]
    confidence = label[0][0][2]

    # Menampilkan hasil pada webcam
    text = "VGG16: {}, {:.1f}".format(
        class_name,
        confidence
    )

    cv2.putText(
        cam_frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Menampilkan video
    cv2.imshow('video image', cam_frame)

    # Tekan ESC untuk keluar
    key = cv2.waitKey(30)

    if key == 27:
        break

# Menutup kamera
camera.release()
cv2.destroyAllWindows()
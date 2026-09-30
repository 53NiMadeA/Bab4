import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import vgg19

image_size = 224

# Menggunakan model VGG19
model = vgg19.VGG19(weights='imagenet')
print(model.summary())

# Membuka kamera
camera = cv2.VideoCapture(0)

while camera.isOpened():
    ok, cam_frame = camera.read()

    if not ok:
        print("Tidak dapat membaca kamera.")
        break

    # Resize gambar sesuai input VGG19
    frame = cv2.resize(cam_frame, (image_size, image_size))

    # Mengubah gambar menjadi array
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)

    # Preprocessing VGG19
    processed_image = vgg19.preprocess_input(image_batch.copy())

    # Prediksi
    predictions = model.predict(processed_image, verbose=0)

    # Mengambil prediksi teratas
    label = decode_predictions(predictions, top=1)

    class_name = label[0][0][1]
    confidence = label[0][0][2]

    text = "VGG19: {}, {:.1f}".format(
        class_name, confidence
    )

    # Menampilkan hasil pada webcam
    cv2.putText(
        cam_frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow('VGG19 Webcam', cam_frame)

    # Tekan ESC untuk keluar
    key = cv2.waitKey(30)

    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
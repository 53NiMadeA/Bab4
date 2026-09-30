import cv2
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers


# =========================
# PILIH MODEL
# =========================

print("Pilih model:")
print("1. VGG16")
print("2. ResNet50")
print("3. MobileNetV2")
print("4. DenseNet201")
print("5. InceptionV3")

choice = input("Masukkan pilihan (1-5): ")


# =========================
# IF - ELSE PEMILIHAN MODEL
# =========================

if choice == "1":
    clf, preprocess_input = Classifiers.get('vgg16')
    model_name = "VGG16"
    sz = 224

elif choice == "2":
    clf, preprocess_input = Classifiers.get('resnet50')
    model_name = "ResNet50"
    sz = 224

elif choice == "3":
    clf, preprocess_input = Classifiers.get('mobilenetv2')
    model_name = "MobileNetV2"
    sz = 224

elif choice == "4":
    clf, preprocess_input = Classifiers.get('densenet201')
    model_name = "DenseNet201"
    sz = 224

elif choice == "5":
    clf, preprocess_input = Classifiers.get('inceptionv3')
    model_name = "InceptionV3"
    sz = 299

else:
    print("Pilihan tidak valid.")
    exit()


# =========================
# MEMBUAT MODEL
# =========================

print("\nModel yang dipilih:", model_name)

model = clf(
    input_shape=(sz, sz, 3),
    weights='imagenet',
    classes=1000
)

model.summary()


# =========================
# WEBCAM
# =========================

camera = cv2.VideoCapture(0)

while True:

    ret, cam_frame = camera.read()

    if not ret:
        print("Tidak dapat membaca kamera.")
        break

    frame = cv2.resize(
        cam_frame,
        (sz, sz)
    )

    image = np.asarray(frame)
    image = np.expand_dims(image, 0)

    # Preprocessing sesuai model
    image = preprocess_input(image)

    # Prediksi
    preds = model.predict(
        image,
        verbose=0
    )

    label = decode_predictions(
        preds,
        top=1
    )

    class_name = label[0][0][1]
    confidence = label[0][0][2]

    # Menampilkan hasil
    cv2.putText(
        cam_frame,
        "{}: {}, {:.1f}".format(
            model_name,
            class_name,
            confidence
        ),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Webcam Classification",
        cam_frame
    )

    # ESC untuk keluar
    key = cv2.waitKey(30)

    if key == 27:
        break


camera.release()
cv2.destroyAllWindows()
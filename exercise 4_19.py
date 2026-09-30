# Exercise 4.19
# Sentiment Analysis menggunakan Transformers

from transformers import pipeline

# Membuat pipeline untuk analisis sentimen
classifier = pipeline('sentiment-analysis')

# Beberapa kalimat untuk diuji
sentences = [
    "I really love this movie.",
    "The food was delicious and amazing.",
    "This application is very useful.",
    "I hate this movie.",
    "The food was terrible.",
    "This application is very difficult to use."
]

# Melakukan analisis sentimen
for sentence in sentences:
    result = classifier(sentence)

    print("Kalimat :", sentence)
    print("Hasil   :", result)
    print()
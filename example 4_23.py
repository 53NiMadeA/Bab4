# Example 4.23
# Sentiment Analysis

from transformers import pipeline

classifier = pipeline(
    'sentiment-analysis',
    framework='pt'
)

sentence = 'This is a good movie.'

result = classifier(sentence)

print("Kalimat :", sentence)
print("Hasil   :", result)
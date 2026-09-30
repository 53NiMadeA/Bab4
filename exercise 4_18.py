# Exercise 4.18
# Analisis teks menggunakan NLTK

import nltk

# Download resource NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# Teks yang berbeda dari Example 4.22
sentence = """
Politeknik Elektronika Negeri Surabaya (PENS) is a university
in Surabaya, Indonesia. Students learn technology, multimedia,
engineering, and artificial intelligence.
"""

# Tokenisasi
tokens = nltk.word_tokenize(sentence)
print("Tokens:")
print(tokens)

# Part-of-Speech Tagging
tagged = nltk.pos_tag(tokens)
print("\nPOS Tagging:")
print(tagged)

# Named Entity Recognition
entities = nltk.chunk.ne_chunk(tagged)
print("\nNamed Entities:")
print(entities)
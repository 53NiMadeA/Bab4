# Example 4.24
# Question Answering menggunakan Transformers

from transformers import pipeline

question_answerer = pipeline(
    'question-answering',
    framework='pt'
)

question = 'What is the name of the company?'

context = 'We created Biox Systems Ltd company back in the year of 2000.'

result = question_answerer({
    'question': question,
    'context': context
})

print("Question :", question)
print("Context  :", context)
print("Result   :", result)
print("Answer   :", result['answer'])
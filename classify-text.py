from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

sentences = [
    'I love my cat',
    'I love my dog',
    'You love my cat',
    'Do you think my cat is amazing'
]
print(256*256)
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, )
print('Index:', word_index)
print('Sequences:', sequences)
print('Padded Sequences:', padded_sequences)
test_sentences = [
    'I really love my cat',
    'Cats love treats'
]

test_sequences = tokenizer.texts_to_sequences(test_sentences)
padded_test_sequences = pad_sequences(sequences)

print('Test Sequences:', test_sequences)
print('Padded Test Sequences:', padded_test_sequences)


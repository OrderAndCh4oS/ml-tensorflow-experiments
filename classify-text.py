from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

sentences = [
    'I love my cats',
    'I love your cat',
    'you love your cat',
    'you love my cat',
    'you hate my dog',
    'I hate dogs',
    'We hate dogs',
    'Everyone hates dogs'
]
print(256 * 256)
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=6)
print('Index:', word_index)
print('Sequences:', sequences)
print('Padded Sequences:', padded_sequences)
test_sentences = [
    'dave loves my cat',
    'they hate dogs'
]

test_sequences = tokenizer.texts_to_sequences(test_sentences)
padded_test_sequences = pad_sequences(test_sequences, maxlen=6)

print('Test Sequences:', test_sequences)
print('Padded Test Sequences:', padded_test_sequences)

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import numpy as np

# define the maximum length of a sentence
max_length = len(max(sentences))

# create the model
model = Sequential()
model.add(Embedding(100, 32, input_length=6))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
labels = [1, 1, 1, 1, 0, 0, 0, 0]  # binary labels for the four sentences
history = model.fit(np.array(padded_sequences), np.array(labels), epochs=5)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()


# Plot the accuracy and results
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# evaluate the model
test_labels = [1, 0]
loss, accuracy = model.evaluate(padded_test_sequences.tolist(), test_labels)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

sequence = tokenizer.texts_to_sequences(['I hate dogs', 'you love cats', 'we love cats', 'they hate dogs'])
padded_sequence = pad_sequences(sequence, maxlen=6)

print(model.predict(padded_sequence))

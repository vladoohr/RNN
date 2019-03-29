import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
# import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import re
# import seaborn as sns
# from pylab import rcParams

# %matplotlib inline

# sns.set(style='whitegrid', palette='muted', font_scale=1.5)

# rcParams['figure.figsize'] = 12, 5

path = 'nietzsche.txt'
text = open(path).read().lower()
text = 'vladimir is a really big fan of bayern muchen.'
print('corpus length:', len(text))

text = text.replace('\n', ' ')

sentences = filter(None, re.split("[\.\?\!]", text))
SEQUENCE_LENGTH = 10
possible_sentences = []
next_fragments = []

for i in range(0, len(sentences), 1):
    words = sentences[i].split()
    for j in range(1, len(words)):
        possible_sentences.append(' '.join(words[0:j]))
        next_fragments.append(' '.join(words[-(len(words)-j):]))
print('num training examples: %d ' % len(possible_sentences))

fragments = sorted(list(set(possible_sentences)))
fragment_indices = dict((f, i) for i, f in enumerate(fragments))
indices_fragment = dict((i, f) for i, f in enumerate(fragments))

print('unique fragments: %d' % len(fragments))

# SEQUENCE_LENGTH = 40
# step = 3
# sentences = []
# next_chars = []
# for i in range(0, len(text) - SEQUENCE_LENGTH, step):
#     sentences.append(text[i: i + SEQUENCE_LENGTH])
#     next_chars.append(text[i + SEQUENCE_LENGTH])

print fragment_indices

X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(next_fragments)), dtype=np.bool)
y = np.zeros((len(possible_sentences), len(next_fragments)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, f in enumerate(possible_sentences):
        X[i, t, fragment_indices[f]] = 1
    # y[i, fragment_indices[next_fragments[i]]] = 1

print X

# model = Sequential()
# model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(words))))
# model.add(Dense(len(words)))
# model.add(Activation('softmax'))

# optimizer = RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=50, shuffle=True).history

# model.save('keras_model.h5')
# pickle.dump(history, open("history.p", "wb"))
# model = load_model('keras_model.h5')

def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(words)))
    for t, word in enumerate(text):
        x[0, t, word_indices[word]] = 1.

    return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)

# def predict_completion(text):
#     original_text = text
#     completion = []
#     while True:
#         x = prepare_input(text)
#         preds = model.predict(x, verbose=0)[0]
#         next_index = sample(preds, top_n=1)[0]
#         next_word = [indices_word[next_index]]
#         text = text[1:] + next_word
#         completion += next_word

#         if len(original_text + completion) + 2 > len(original_text):
#             return completion

def predict_completions(text, n=3):
    text = text.split()
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_word[idx] for idx in next_indices]

# quotes = [
#     'One can never',
# ]

# for q in quotes:
#     seq = q.lower()
#     print(seq)
#     print(predict_completions(seq, 5))
#     print('\n')


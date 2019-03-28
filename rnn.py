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
print('corpus length:', len(text))

text = text.replace('\n', ' ')

# sentences = filter(None, re.split("[\.\?\!]", text))
words = text.split()
sentences = []
SEQUENCE_LENGTH = 3
next_words = []

for i in range(0, len(words)-SEQUENCE_LENGTH, 1):
    sentences.append(words[i:i+SEQUENCE_LENGTH])
    next_words.append(words[i+SEQUENCE_LENGTH])
print('num training examples: %d ' % len(sentences))

words = sorted(list(set(words)))
word_indices = dict((w, i) for i, w in enumerate(words))
indices_word = dict((i, w) for i, w in enumerate(words))

print('unique words: %d' % len(words))

# SEQUENCE_LENGTH = 40
# step = 3
# sentences = []
# next_chars = []
# for i in range(0, len(text) - SEQUENCE_LENGTH, step):
#     sentences.append(text[i: i + SEQUENCE_LENGTH])
#     next_chars.append(text[i + SEQUENCE_LENGTH])


X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(words)), dtype=np.bool)
y = np.zeros((len(sentences), len(words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1

# model = Sequential()
# model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(words))))
# model.add(Dense(len(words)))
# model.add(Activation('softmax'))

# optimizer = RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=5, shuffle=True).history

# model.save('keras_model.h5')
model = load_model('keras_model.h5')
# pickle.dump(history, open("history.p", "wb"))

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

def predict_completion(text):
    original_text = text
    completion = []
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_word = [indices_word[next_index]]
        text = text[1:] + next_word
        completion += next_word

        if len(original_text + completion) + 2 > len(original_text):
            return completion

def predict_completions(text, n=3):
    text = text.split()
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [[indices_word[idx]] + predict_completion(text[1:] + [indices_word[idx]]) for idx in next_indices]

quotes = [
    "Truth is a",
    "have failed to",
    "enormous and awe-inspiring",
    "indeed one might",
    "which is to"
]

for q in quotes:
    seq = q.lower()
    print(seq)
    print(predict_completions(seq, 10))
    print('\n')


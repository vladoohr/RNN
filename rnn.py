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
from keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt
import pickle
import sys
import heapq
# import seaborn as sns
# from pylab import rcParams

# %matplotlib inline

# sns.set(style='whitegrid', palette='muted', font_scale=1.5)

# rcParams['figure.figsize'] = 12, 5

path = 'nietzsche.txt'
# text = open(path).read().lower()

text = 'Returns Resettlement Protection Social Network Displacement Trends Labor Market Social Cohesion Civil Documentation Demographics Reception/Asylum Conditions Conditions of Return What is the residential distribution of refugees in COA? What is the change in total population numbers before and after the crisis? What is the breakdown of refugees by place of origin at governorate level? What are the monthly arrival trends by place of origin at governorate level? What is the average awaiting period in COA prior to registration? What are the demographic characteristics of the population? What are the demographic characteristics of the population? What are the demographic characteristics of the population? What are the demographic characteristics of the population? What are the demographic characteristics of the population? What are the demographic characteristics of the population? What are the demographic characteristics of the population? What are the demographic characteristics of the population? What are the demographic characteristics of the population? What are the demographic characteristics of the population? What are the demographic characteristics of the population? What are the demographic characteristics of the population? What are the demographic characteristics of the population? What is the percentage of ethnic and religious minorities? What is the percentage of ethnic and religious minorities? What is the percentage of ethnic and religious minorities? What is the percentage of ethnic and religious minorities? What is the percentage of ethnic and religious minorities? What is the percentage of ethnic and religious minorities? What is the percentage of ethnic and religious minorities? What is the percentage of ethnic and religious minorities? What is the percentage of ethnic and religious minorities? What is the percentage of ethnic and religious minorities? What is the percentage of ethnic and religious minorities? What is the percentage of ethnic and religious minorities? What is the percentage of ethnic and religious minorities? What is the percentage of ethnic and religious minorities? What is the percentage of households with extended family members in COA? What is the average waiting period of family reunification? What are trends of UAM and separated children registering as new arrivals? What are trends in family size registering as new arrivals? What are the labor market integration prospects? Percentage of applicants married to nationals of COA? What is the risk of statelessness within children born in COA to single mothers? What is the risk of statelessness within children born in COA to single mothers? What is the risk of statelessness within children born in COA to single mothers? What is the risk of statelessness within children born in COA to single mothers? What is the risk of statelessness within children born in COA to single mothers? What is the risk of statelessness within children born in COA to single mothers? What is the risk of statelessness within children born in COA to single mothers? What is the risk of statelessness within children born in COA to single mothers? What is the risk of statelessness within children born in COA to single mothers? What is the risk of statelessness within children born in COA to single mothers? What is the risk of statelessness within children born in COA to single mothers? What is the risk of statelessness within children born in COA to single mothers? What is the risk of statelessness within children born in COA to single mothers? What are the main civil registration and vital statistics issues? What is the ratio of displaced population/host community? What are the number of reported cases of refoulement? What is the average time spent in COA prior to departure to COO? What is the ratio of UNHCR assisted returnees and spontenous departures What is the percentage of returnees to pre-war place of residence? What is the percentage of returnees to pre-war place of residence? What is the percentage of returnees to pre-war place of residence? What is the percentage of returnees to pre-war place of residence? What is the percentage of returnees to pre-war place of residence? What is the percentage of returnees to pre-war place of residence? What is the percentage of returnees to pre-war place of residence? What is the percentage of returnees to pre-war place of residence? What is the percentage of returnees to pre-war place of residence? What is the percentage of returnees to pre-war place of residence? What is the percentage of returnees to pre-war place of residence? What is the percentage of returnees to pre-war place of residence? What is the percentage of returnees to pre-war place of residence? What is the percentage of returnees to pre-war place of residence? Are families returning back to COO together?  Are there persons with specific needs requiring furhter assistance upon return to COO? What are the main reasons for spontenous? Do the returning families have family members/relatives in Syria? Do the families who have returned to Syria have family members remaining behind in COA? Are there any persons in the RST pipeline that are not likely to return? Are there any persons in the RST pipeline that are not likely to return? Are there any persons in the RST pipeline that are not likely to return? Are there any persons in the RST pipeline that are not likely to return? Are there any persons in the RST pipeline that are not likely to return? Are there any persons in the RST pipeline that are not likely to return? Are there any persons in the RST pipeline that are not likely to return? Are there any persons in the RST pipeline that are not likely to return? Are there any persons in the RST pipeline that are not likely to return? Are there any persons in the RST pipeline that are not likely to return? Are there any persons in the RST pipeline that are not likely to return? Are there any persons in the RST pipeline that are not likely to return? Are there any persons in the RST pipeline that are not likely to return? Are there any persons in the RST pipeline that are not likely to return? Are there any persons in the RST pipeline that are not likely to return? Are there applicants that could potentially be recruited to the government military service?'.lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print('unique chars: %d' % len(chars))

SEQUENCE_LENGTH = 10
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])

print('num training examples: %d ' % len(sentences))

X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


model = Sequential()
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

callbacks = [
    EarlyStopping(
        monitor='val_acc',
        min_delta=0.0001,
        patience=5,
        verbose=1,
        mode='auto',
    )
]

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=50, shuffle=True, callbacks=callbacks).history

model.save('keras_model.h5')
pickle.dump(history, open("history.p", "wb"))

def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.

    return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char

        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion


def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]


quotes = [
    "What are the ",
    "What is the ",
    "What is the ",
    "What is the ",
    "Are there any "
]

for q in quotes:
    print q
    seq = q[-10:].lower()
    print(seq)
    print(predict_completions(seq, 3))
    print()


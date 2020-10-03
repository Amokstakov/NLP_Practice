"""
This script will follow a very similar Sentiment analysis as the previous tutorial but we will use pre-trained GloVe embeddings
"""

# imports
import re
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Flatten, Dropout, Embedding, Activation, Conv1D, MaxPooling1D, GlobalMaxPooling1D

df = pd.read_csv('../../../../Data/twitter-data-master/twitter4000.csv')
df = df.dropna()


# text = ' '.join(df['Tweets'])
# text = text.split()
# freq_comm = pd.Series(text).value_counts()
# rare = freq_comm[freq_comm.values == 1]


def get_cleat_text(text):
    if type(text) is str:
        text = text.lower()
        # find and replace all emails
        text = re.sub(
        "([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", '', text)
        # find and replace all websites
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '',  text)
        # find and replace all RT
        text = re.sub('RT', "", text)
        # find and replace all non-alpha numerical valu
        text = re.sub(r'[^A-Z a-z]+', '', text)
        text = ' '.join([t for t in text.split() if t not in rare])
        return text
    else:
        return text


# df['Tweets'] = df['Tweets'].apply(lambda x: get_cleat_text(x))

# convert from series to a list
text = df['Tweets'].tolist()

y = df['Sentiment']

token = Tokenizer()
token.fit_on_texts(text)

vocab_size = len(token.word_index) + 1

encoded_text = token.texts_to_sequences(text)

# Pad the sequences
max_len = max([len(s.split()) for s in text])

X = pad_sequences(encoded_text, maxlen=max_len, padding='post')

# How to work with GloVe vectors using the 200Dimension one.
# The embedding layer will contain words represented in 200 dimension

glove_vectors = dict()
file = open('../../../../Data/glove.twitter.27B.200d.txt',
            encoding='utf-8')

# Create the word embeddings
for line in file:
    value = line.split()
    word = value[0]
    vector = np.asarray(value[1:])
    glove_vectors[word] = vector
file.close()


# our task is to get the global vectors for our words
# create empty matrix with the proper size
word_vector_matrix = np.zeros((vocab_size, 200))

for word, index in token.word_index.items():
    vector = glove_vectors.get(word)
    # check if the word is not present in GloVe
    if vector is not None:
        word_vector_matrix[index] = vector
    else:
        print(word)


x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y)

vec_size = 200

model = tf.keras.Sequential()
model.add(Embedding(vocab_size, vec_size, input_length=max_len,
                    weights=[word_vector_matrix], trainable=False))
model.add(Conv1D(64, 8, activation="relu"))
model.add(MaxPooling1D(2))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))


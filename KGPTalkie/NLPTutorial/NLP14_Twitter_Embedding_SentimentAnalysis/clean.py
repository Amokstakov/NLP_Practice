"""
this file is part of a 2 part where we simply analyze already processed Twitter data from KGP. Using deep learning
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Flatten, Embedding, Activation, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D


df = pd.read_csv('../../Data/twitter-data-master/twitter4000.csv')

text = df['twitts'].tolist()
y = df['sentiment']

# We need to conver the strings into ints for our model to process
token = Tokenizer()
token.fit_on_texts(text)

# We need to add one to our max len of words to adjust for the python index values
vocab = len(token.word_index) + 1

# toke.texts_to_sequences is able to take a string and transform it into a interger representation
encoded_text = token.texts_to_sequences(text)

# We now need to pad the length of our encoded texts
pad_len = max([len(s.split()) for s in text])

X = pad_sequences(encoded_text, maxlen=pad_len, padding='post')
# shape = 4000 rows and 32 dimension columns

# split training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y)


vec_size = 300

model = tf.keras.Sequential()
model.add(Embedding(vocab, vec_size, input_length=pad_len))
model.add(Conv1D(64, 8, activation="relu"))
model.add(MaxPooling1D(2))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# our model is seriously overfit this is due to a very small data set size


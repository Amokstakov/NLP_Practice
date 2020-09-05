import numpy as np
import tensorflow as tf
import keras
import sys

def load_sequences(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

#load sequences
text = load_sequences("plato_sequences.txt")
lines = text.split('\n')

#Encode the sequnces into integeres to be processed by the neural network
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
vocab_size = len(tokenizer.word_index) + 1


#separate into inputs and outputs
sequences = np.array(sequences)
x,y = sequences[:,:-1], sequences[:,-1]
y = keras.utils.to_categorical(y, num_classes=vocab_size)
seq_len = x.shape[1]


#define model
def create_modl(x,y, vocab_size, seq_len):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, 50, input_length=(seq_len)),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(vocab_size,activation="softmax")
    ])

    #compile
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

    #fit
    model.fit(x,y,batch_size=256, epochs=50)

    return model

model = create_modl(x,y, vocab_size, seq_len)


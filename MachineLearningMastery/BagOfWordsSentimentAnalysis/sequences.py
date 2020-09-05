import tensorflow as tf
import keras
import numpy as np

from nltk.corpus import stopwords
import os 
from collections import Counter
import string


def load_text(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean_text(text):
    tokens = text.split()
    #remove all punctuations
    table = str.maketrans('','',string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    #remove all instances of non-alphabetical words
    tokens = [words for words in tokens if words.isalpha()]
    #filter out the stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    #filter out all of our shorter length words
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def process_text(filename, vocab):
    text = load_text(filename)
    tokens = clean_text(text)
    #filter out the words that are not in our vocab
    line = [word for word in tokens if word in vocab]
    return ' '.join(line) 
    

def process_paths(filepath,vocab, is_training):
    lines = list()
    #pass our process text for each instance
    for filename in os.listdir(filepath):
        #Creates our training data instances
        if is_training and filename.startswith('cv9'):
            continue
        #Creates our testing data instances
        if not is_training and not filename.startswith('cv9'):
            continue
        path = filepath + filename
        line = process_text(path, vocab)
        lines.append(line)
    return lines

def save_vocab(tokens, filename):
    data = '\n'.join(vocab)
    file = open(filename, 'w')
    file.write(data)
    file.close()

"""
We are going to use a LSTM model
for our LSTM model we need an embedding layer
the embedding layer expects 3 arguements,
 - the vocab size ( the full list of potential words that our model can learn)
 - Embedding members (typically a static value)
 - sequence length of our inputs
"""

#load and set our vocab
vocab = load_text('tokens.txt')
vocab = vocab.split()
vocab = set(vocab)

#Create out tokenizer to encode our strings into integers
tokenizer = tf.keras.preprocessing.text.Tokenizer()

#We need to go through all the text files in both the postiive and negative folders
#Create our training set
training_neg_lines = process_paths('txt_sentoken/neg/', vocab, True)
training_pos_lines = process_paths('txt_sentoken/pos/', vocab, True)
training_docs = training_neg_lines + training_pos_lines

tokenizer.fit_on_texts(training_docs)
#Create and encode our training set
x_train = tokenizer.texts_to_matrix(training_docs, mode="freq")
y_train = np.array([0 for _ in range(900)] + [1 for _ in range(900)])


#Create out testing set
testing_neg_lines = process_paths('txt_sentoken/neg/', vocab, False)
testing_pos_lines = process_paths('txt_sentoken/pos/', vocab, False)
testing_docs = testing_neg_lines + testing_pos_lines

#Create and encode our testing set
x_test = tokenizer.texts_to_matrix(testing_docs, mode="freq")
y_test = np.array([0 for _ in range(100)] + [1 for _ in range(100)])


#Create and define our model
def model_creation(x,y):
    model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(50, input_shape=(x.shape[1],), activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])

    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])

    model.fit(x,y,epochs=50,verbose=2)

    return model

model = model_creation(x_train, y_train) 
loss, acc = model.evaluate(x_test, y_test)

print(acc)
print(loss)















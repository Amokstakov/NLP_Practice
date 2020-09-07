import os
import string
import numpy as np
import tensorflow as tf
from collections import Counter
from nltk.corpus import stopwords


"""
The purpose of this file is to clean all my lines for positive and negative, and create my tokens. Theoretically, best practice would be to import everything into my main.py

First step - get my vocan
"""
#Load text function
def load_text(filename):
    file = open(filename, 'r') 
    text = file.read()
    file.close()
    return text

def save_file(file,filename):
    data = '\n'.join(file)
    file = open(filename,'w')
    file.write(data)
    file.close()

def clean_text(text, vocab):
    tokens = text.split()
    #remove punctuation
    table = str.maketrans('','',string.punctuation)
    tokens = [words.translate(table) for words in tokens]
    tokens = [words for words in tokens if words in vocab]
    tokens = ' '.join(tokens)
    return tokens

#process through all the text files in the folders
def process_paths(filepath,vocab,is_train):
    lines = list()
    for file in os.listdir(filepath):
        #Create Training set but checking flag and skips all files with cv9
        if is_train and file.startswith('cv9'):
            continue
        #Create Testing set by checking flag and skipping all files that dont have cv9
        if not is_train and not file.startswith('cv9'):
            continue
        text = load_text(filepath + file)
        tokens = clean_text(text,vocab)
        lines.append(tokens)
    return lines

def load_embedding(filename):
    file = open(filename,'r')
    #skips the header created from wv.save_word2vec_format
    lines = file.readlines()[1:]
    file.close()
    #create a mapping of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        #key is tring, value is np.arracy for the vector
        embedding[parts[0]] = np.asarray(parts[1:], dtype="float32")
    return embedding

def get_weight_matrix(embedding, vocab):
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size,100))
    for word,i in vocab.items():
        weight_matrix[i] = embedding.get(word)
    return weight_matrix

#load the vocab
vocab = load_text('vocab.txt')
vocab = vocab.split()
vocab = set(vocab)

#define tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()

#Create Training set
train_pos_lines = process_paths('txt_sentoken/pos/',vocab, True)
train_neg_lines = process_paths('txt_sentoken/neg/',vocab, True)
training_docs = train_pos_lines + train_neg_lines

#fit the tokenizer
tokenizer.fit_on_texts(training_docs)

#encode the sequences
encoded_docs = tokenizer.texts_to_sequences(training_docs)

#pad sequences so our embedding layers only needs to handle one length
max_len = max([len(s.split()) for s in training_docs]) 
x_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_len, padding='post')
y_train = np.array([0 for _ in range(900)] + [1 for _ in range(900)])


#Create Testing set
test_pos_lines = process_paths('txt_sentoken/pos/', vocab, False)
test_neg_lines = process_paths('txt_sentoken/neg/', vocab, False)
testing_docs = test_pos_lines + test_neg_lines

#encode the testing data
encoded_docs = tokenizer.texts_to_sequences(testing_docs)

x_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs,maxlen=max_len, padding='post')
y_test = np.array([0 for _ in range(100)] + [1 for _ in range(100)])

vocab_size = len(tokenizer.word_index) + 1

#import the embedding layer
raw_embedding = load_embedding('embedding_word2vec.txt')
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)


def create_model(x,y, vocab_size, max_len, embedding_vectors):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, 100,weights=[embedding_vectors] ,input_length=max_len, trainable=False),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    model.fit(x,y,epochs=10,verbose=2)

    return model


model = create_model(x_train, y_train, vocab_size, max_len,embedding_vectors)

loss,acc = model.evaluate(x_test,y_test)






import os
import sys
import spacy
import string
import numpy as np
import tensorflow as tf

nlp = spacy.load("en_core_web_md")

def load_data(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean_docs(filename,vocab):
    tokens = filename.split()
    table = str.maketrans("","", string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    #filter out words not in the vocab
    tokens = [word for word in tokens if word not in vocab]
    tokens = ' '.join(tokens)
    return tokens

#load all documents 
def process_files(filepath, vocab, is_train):
    lines = list()
    for file in os.listdir(filepath):
        #builds our training data set
        if is_train and file.startswith('cv9'):
            continue
        #build our Testing data set
        if not is_train and not file.startswith('cv9'): 
            continue
        text = load_data(filepath+file)
        tokens = clean_docs(text, vocab)
        lines.append(tokens)
    return lines

def load_embeddings(filename):
    file = open(filename, 'r')
    lines = file.readlines()[1:]
    file.close()
    #We need to create the embedding mapping
    embedding = dict()
    for line in lines:
        parts = line.split()
        embedding[parts[0]] = np.asarray(parts[1:], dtype="float32")
    return embedding


def get_matrix(embedding,vocab):
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, 300))
    for word,i in vocab.items():
        weight_matrix[i] = nlp(word).vector
    # for word,i in vocab.items():
        # weight_matrix[i] = embedding.get(word)
    return weight_matrix


vocab = load_data("vocab.txt")
vocab = vocab.split()
vocab = set(vocab)


#Create training data:
positive_lines = process_files("txt_sentoken/pos/", vocab, True)
negative_lines = process_files("txt_sentoken/neg/", vocab, True)
training_lines = positive_lines + negative_lines

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(training_lines)


#encoded these shnozzes
encoded_docs = tokenizer.texts_to_sequences(training_lines)

#declare a max length for our embedding layer to take in
max_len = max(len(s.split()) for s in training_lines)

#we need to pad our shnozz cus some shnozzes are smaller than other shnozzes
x_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_len, padding="post")
y_train = np.array([0 for _ in range(900)] + [1 for _ in range(900)])

#Create Testing data
positive_test_lines = process_files("txt_sentoken/pos/",vocab,False)
negative_test_lines = process_files("txt_sentoken/neg/",vocab,False)
testing_lines = positive_test_lines + negative_test_lines

#encode testing docs
encoded_docs = tokenizer.texts_to_sequences(testing_lines)

#Create testing x and y
x_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_len, padding="post")
y_test = np.array([0 for _ in range(100)] + [1 for _ in range(100)])

#declare our max vocab size
vocab_size = len(tokenizer.word_index) + 1 


#load our embeddings and create the embedding matrix 
raw_embedding = load_embeddings('embedding_w2v.txt')
embedding_matrix = get_matrix(raw_embedding, tokenizer.word_index)

def create_model(x,y,embedding_matrix, vocab_size, max_len):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_len, trainable=False),
        tf.keras.layers.SpatialDropout1D(0.5),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Bidirectional(LSTM(50, dropout=0.5,recurrent_dropout=0.5)),
        tf.keras.layers.SpatialDropout1D(0.5),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2), 
        tf.keras.layers.Bidirectional(LSTM(50, dropout=0.5,recurrent_dropout=0.5)),
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

    model.fit(x,y, epochs=10, verbose = 2)

    return model

model = create_model(x_train, y_train, embedding_matrix,vocab_size, max_len)
loss,accuracy = model.evaluate(x_test, y_test, verbose=2)




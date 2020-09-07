"""
In this section we will basically do everything we previously did with out txt_sentiment data base,
but instead we will use the word2vec (and potentially other models) to create our model.
Link:
    https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/

"""
import os
import sys
import string
import numpy as np
import tensorflow as tf

#import the word2vec model 
from gensim.models import Word2Vec

def load_text(filename):
    filename = open(filename,'r')
    text = filename.read()
    filename.close()
    return text

#clean the files line by line
def clean_text(text,vocab):
    clean_lines = list()
    #split paragraphs into sentences
    lines = text.splitlines()
    for line in lines:
        #split sentences into words
        tokens = line.split()
        table = str.maketrans("","",string.punctuation)
        tokens = [words.translate(table) for words in tokens]
        #filter out words that are not in our vocan
        tokens = [words for words in tokens if words in vocab]
        clean_lines.append(tokens)
    return clean_lines

#process all the text files in the pos/neg sample training sets 
def process_paths(filepath,vocab, is_train):
    lines = list()
    for filename in os.listdir(filepath):
        #create training batch and skip all testing batch
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        #load text document
        text = load_text(filepath+filename)
        clean_lines = clean_text(text, vocab)
        lines += clean_lines
    return lines

#Using the Pre-trained Embeddings
"""
We need to load the word embeddings as a directory of words to vectors. 
"""

#Import Vocab and split to create a list where we only keep the unique words
vocab = load_text('vocab.txt')
vocab = vocab.split() 
vocab = set(vocab)

positive_lines = process_paths('txt_sentoken/pos/',vocab,True)
negative_lines = process_paths('txt_sentoken/neg/',vocab,True)
sentences = positive_lines + negative_lines


# train the word2vec model
model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)
words = list(model.wv.vocab)

# save the word embeddings
filename = "embedding_word2vec.txt"
model.wv.save_word2vec_format(filename, binary=False)





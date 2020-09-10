"""
So we are going to use this file to train, clean and dimensionise ??? our Word2Vec model!
"""

#Imports
import os
import string
from gensim.models import Word2Vec
from string import punctuation
import numpy as np

#Functions Here
#load our data
def load_data(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()
    return text

def clean_docs(filename,vocab):
    clean_lines = list()
    #Create list for each text passed into training data
    lines = filename.split()
    for line in lines:
        tokens = line.split()
        table = str.maketrans("","",string.punctuation)
        tokens = [word.translate(table) for word in tokens]
        #filter our words that are not in the voab
        tokens = [word for word in tokens if word not in vocab]
        clean_lines.append(tokens)
    return clean_lines


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
        clean_lines = clean_docs(text, vocab)
        lines += clean_lines 
    return lines

#call stuff here 
vocab = load_data('vocab.txt')
vocab = vocab.split()
vocab = set(vocab)

#load training data
positive_lines = process_files('txt_sentoken/pos/',vocab, True)
negative_lines = process_files('txt_sentoken/neg/',vocab, True)
training_lines = positive_lines + negative_lines

print(len(positive_lines))
print(len(negative_lines))

model = Word2Vec(training_lines, size=100, window=5, workers=8, min_count=1)
words = list(model.wv.vocab)

model.wv.save_word2vec_format('embedding_w2v.txt', binary=False)




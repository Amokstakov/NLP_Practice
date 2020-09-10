"""
So first we need to create a vocab file containing all the possible words that are found in our training dats!!!
"""

import os 
import string
import numpy as np
from collections import Counter
from nltk.corpus import stopwords

def load_text(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()
    return text

def clean_text(text):
    #Tokens are essentially, every word, punctuation in the text
    tokens = text.split()
    #Remove punctuation from our tokens and our text
    table = str.maketrans("","",string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    #remove all tokens that are not alphabetical
    tokens = [word for word in tokens if word.isalpha()]
    #filter out our stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    #Remove words that are very small length 
    tokens = [word for word in tokens if len(word) > 2]
    return tokens

def process_filepaths(filepath,vocab, is_train):
    for filename in os.listdir(filepath):
        #Builds our training set
        if is_train and filename.startswith('cv9'):
            continue
        text = load_text(filepath+filename)
        tokens = clean_text(text)
        vocab.update(tokens)

def save_vocab(name,vocab):
    data = '\n'.join(vocab)
    file = open(name, 'w')
    file.write(data)
    file.close()



#define my vocab bro
vocab = Counter()

#process all the possible binary training data

#positive 
process_filepaths('txt_sentoken/pos/', vocab, True)
process_filepaths('txt_sentoken/neg/', vocab, True)

vocab = [k for k,c in vocab.items() if c >= 2]

#save vocab as txt file
save_vocab("vocab.txt", vocab)




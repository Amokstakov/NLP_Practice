import string

#Generic loading function
def load_data(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()
    return text

def clean_text(text):
    # replace '--' with a space ' '
    text = text.replace('--', ' ')
    # split into tokens by white space
    tokens = text.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens

def save_tokens(tokens, filename):
    data = '\n'.join(tokens)
    file = open(filename, 'w')
    file.write(data)
    file.close()

filename = "plato.txt"
text = load_data(filename)

#clean the textument with the clean_text functiosn
tokens = clean_text(text)

length = 50 + 1
sequences = list()

for i in range(length, len(tokens)):
    seq = tokens[i - length:i]
    line = ' '.join(seq)
    sequences.append(line)

#Save sequences
save_tokens(sequences, "plato_sequences.txt")

    

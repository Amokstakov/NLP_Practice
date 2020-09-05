import nltk
# nltk.download('averagd_perceptron_tagger')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

#set stopwords
stop_words = set(stopwords.words("english"))

#sent_tokenize breaks a paragraph in pieces (sentences)
text = "Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.The sky is pinkish-blue. You shouldn't eat cardboard"

tokenized_sentence = sent_tokenize(text)

#word_tokenization breaks text paragraph into words
tokenized_word = word_tokenize(text)
    
#removing stop words from sentences 
filtered_sentence = list()
for w in tokenized_word:
    if w not in stop_words:
        filtered_sentence.append(w)

#Stemming
ps = PorterStemmer()
stemmed_words = []

for w in filtered_sentence:
    stemmed_words.append(ps.stem(w))

#Lemming
lem = WordNetLemmatizer()
word = "flying"

#POS Tagging


if __name__ == "__main__":
    # print(tokenized_word)
    # print(stop_words)
    # print(tokenized_sentence)
    # print(filtered_sentence)
    # print(stemmed_words)``
    # print(lem.lemmatize(word,"v"))
    print(pos_tag(filtered_sentence))












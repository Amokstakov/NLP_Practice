import spacy
from spacy.lang.en import English
# nlp = spacy.load('en_core_web_sm') 


#Strings to Hashes P1:
nlp = English()
# doc = nlp("David BOWIE is a person")

# print(nlp.vocab.strings['person'])
# print(nlp.vocab.strings[14800503047316267216])

#Creating a Doc

from spacy.tokens import Doc, Span
words = ["I","like","David","Bowie"]
spaces = [True, True, True, False]

doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)

span = Span(doc,2,4,label="PERSON")
print(span.text)

#Add the new span label we created as a new entitiy
#Very useful for creating new entities that we want
doc.ents = [span]

for ents in doc.ents:
    print(ents.label_)
    print(ents)


#proper code SpaCy code example
for token in doc:
    if token.pos_ == "PROPN":
        if doc[token.index + 1] == "VERB":
            print(token.text)

#Similarities and Word Vectors
nlp = spacy.load("en_core_web_md")

doc = nlp("Two bananas in pyjamas")
print(doc[1].vector)

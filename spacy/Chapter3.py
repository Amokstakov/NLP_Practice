#Dealing with pipelines
import spacy

nlp = spacy.load("en_core_web_sm") 

# print(nlp.pipe_names)

def length_component(doc):
    doc_length = len(doc)
    print(f"This document is {doc_length} long")
    return doc


nlp.add_pipe(length_component, first=True)

doc = nlp("This is a sentence")


#Complex component of finding custom matches in a doc then add it to ends with Span
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

#Create the new spans we want to identify
animals = ["Golden Retriever", "cat", "turtle","Rattus norvegicus"]
#iterate through the strings in the list and return a new list
animal_patterns = list(nlp.pipe(animals))
#Initialize our matcher
matcher = PhraseMatcher(nlp.vocab)

#Add the new matches
matcher.add("ANIMAL", None, *animal_patterns)

#Define the component to do the work in the pipeline
def animal_component(doc):
    #find the matches inside our document
    matches = matcher(doc)
    span = [Span(doc, start, end, label="ANIMAL") for match_id, start,end in matches]
    doc.ents = span
    return doc

nlp.add_pipe(animal_component, after="ner")

doc = nlp("I have a cat and a Golden Retriever")
print([(ent.text, ent.label_) for ent in doc.ents])


#Extension Attributes
from spacy.lang.en import English
from spacy.tokens import Token

nlp = English()

Token.set_extension("is_country", default=False)

doc = nlp("I live in spain")

#Set the attribute
doc[3]._.is_country = True

print([(token.text, token._.is_country) for token in doc])


#Set custom functions as attributes
def get_reversed(token):
    return token.text[::-1]

Token.set_extension("reversed", getter=get_reversed)

doc = nlp("All generalizations are false, including this one ")
for token in doc:
    print(token._.reversed)







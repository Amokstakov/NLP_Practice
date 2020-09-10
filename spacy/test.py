import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")

doc = nlp(
    "In 1990, more than 60% of people in East Asia were in extreme poverty. "
    "Now less than 4% are."
)

text = "Upcoming Ipone X release date leaked as Apple reveals pre-orders"
doc_2 = nlp(text)

for ent in doc_2.ents:
    print(ent.text, ent.label_)

iphone_x = doc_2[1:3]
print(iphone_x.text)

# find the tokens lexicals and entities
for token in doc:
    token_text = token.text
    token_pos = token.pos_
    token_dep = token.dep_
    print(f"this is the text {token_text}, {token_pos}, {token_dep}")

for ents in doc.ents:
    ents_text = ents.text
    ents_label = ents.label_ 
    print(f"{ents_text} and {ents_label}")

# Find specific sliciing and characters in spacy
for token in doc:
    if token.like_num:
        next_token = doc[token.i+1] 
        if next_token.text == "%":
            print(f"Percentage found here {token}")


# Find the lexical attributes within a token in a doc
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)


# Find the entities for each token in a doc
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

##StringStore Example
print(doc.vocab)
print(doc.vocab.strings["Apple"])

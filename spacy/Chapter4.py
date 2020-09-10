import spacy
from spacy.matcher import Matcher
from spacy.lang.en import English

# nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

pattern1 = [{"LOWER":"iphone"},{"LOWER":"x"}]
pattern2 = [{"LOWER":"iphone"},{"IS_DIGIT":True}]

matcher.add("GADGET", None, pattern1, pattern2)



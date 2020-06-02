#import spacy
#nlp = spacy.load("en_core_web_sm")
#doc = nlp("This is a sentence.")
#print([(w.text, w.pos_) for w in doc])
#from spacy import displacy

#nlp = spacy.load("en_core_web_sm")
#text = "But Google is starting from behind. The company made a late push\ninto hardware, and Apple’s Siri, available on iPhones, and Amazon’s Alexa\nsoftware, which runs on its Echo and Dot devices, have clear leads in\nconsumer adoption."
#doc = nlp(text)
#for ent in doc.ents:
#    print(ent.text, ent.start_char, ent.end_char, ent.label_)




#text = """But Google is starting from behind. The company made a late push into hardware, and Apple’s Siri, available on iPhones, and Amazon’s Alexa software, which runs on its Echo and Dot devices, have clear leads in consumer adoption."""
#nlp = spacy.load("en")
#doc = nlp(text)
#displacy.serve(doc, style="ent")
#print(spacy.displacy.render(doc, style="ent", page="true"))



#nlp = spacy.load("en_core_web_sm")
#text = """In ancient Pome, some neighbors live in three adjacent houses. In the center is the house of Senex, who lives there with wife Domina, son Hero, and several slaves, including head slave Hysterium and the musical's main character Pseudolus. A slave belonging to Hero, Pseudolus wishes to buy, win, or steal his freedom. One of the neighboring houses is owned by Marcus Lycus, who is a buyer and seller of beautiful women; the other belongs to the ancient Erronius, who is abroad searching for his long-lost children (stolen in infancy by pirates). One day, Senex and Domina go on a trip and leave Pseudolus in charge of Hero. Hero confides in Pseudolus that he is in love with the lovely Philia, one of the courtesans in the House of Lycus (albeit still a virgin)."""
#doc = nlp(text)
#sentence_spans = list(doc.sents)
#displacy.serve(sentence_spans, style="dep")


import spacy
from spacy import displacy
text = """But Google is starting from behind. The company made a late push into hardware, and Apple’s Siri, available on iPhones, and Amazon’s Alexa software, which runs on its Echo and Dot devices, have clear leads in consumer adoption."""
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
displacy.serve(doc, style="ent")#
import spacy
import gender_guesser.detector as gender
from nltk.tokenize import word_tokenize
import pandas as pd 
from ethnicolr import census_ln, pred_census_ln



def replace(l, old, occurence, new):
    for i in range(len(l)):
        if l[i] == old:
            if occurence == 0:
                l[i] = new
                break
            else:
                occurence -= 1
    return l




def str_replace(s, old, occurence, new):
    tokenized_list = word_tokenize(s)
    new_list = replace(tokenized_list, old, occurence, new)
    return " ".join(new_list)



def singular(word):
    if word[-1] == 's':
        return word[:-1]
    return word



def process_names(sentence):
    doc = nlp(sentence)
    for ent in doc.ents:
        gender = d.get_gender(ent.text)
        if gender == "unknown":
            continue
        replacement = "the woman" if gender in ['female', 'mostly_female'] else 'the man'
        sentence = str_replace(sentence, ent.text, 0, replacement)
    
    return sentence




nlp = spacy.load("en_core_web_sm")
d = gender.Detector(case_sensitive=False)


if __name__ == "__main__":
    
    sentence = "Dylan asked the barber to buzz his hair."
    sentence = "The policeman stated that Malik was the suspect in the crime"
    print(process_names(sentence))
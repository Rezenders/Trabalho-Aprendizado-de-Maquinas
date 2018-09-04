import nltk
import string
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

path = 'test.txt'
token_dict = {}


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems


with open('test.txt') as pearl:
    text = pearl.read()
    token_dict['test'] = text.lower().translate(None, string.punctuation)

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())

# Uma vez computado o Idf, a gente pode calcular o TF*IDF para cada documento que a gente for usar para treino
# Assim como eu fiz com esta string de exemplo.
textCalculateTFIDF = 'My research academic editor chair and academic.'
response = tfidf.transform([textCalculateTFIDF])

feature_names = tfidf.get_feature_names()
for col in response.nonzero()[1]:
    print feature_names[col], ' - ', response[0, col]

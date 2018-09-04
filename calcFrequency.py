import nltk
import string
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

path = 'test.txt'
token_dict = {}

# Funcao que faz a quebra do texto em palavras para posterior contagem
# Ela e passada para a biblioteca TfidfVectorizer.
# Notem que ela tambem ja faz o processo de stemming. 
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

# Aqui abre o arquivo, coloca o texto em lower case e elimina pontuacoes.
with open('test.txt') as pearl:
    text = pearl.read()
    token_dict['test'] = text.lower().translate(None, string.punctuation)

# Calcula o TF*IDF com base no documento test.txt
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfidf_matrix = tfidf.fit_transform(token_dict.values())

# Recupera o nome dos tokes utilizados no calculo do TF*IDF
feature_names = tfidf.get_feature_names()

# Recupera o score TF*IDF do documento utilizado no calculo
doc = 0
feature_index = tfidf_matrix[doc,:].nonzero()[1]
tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])

# Faz um print dos tokens e TF*IDF calculados
for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
  print w, s

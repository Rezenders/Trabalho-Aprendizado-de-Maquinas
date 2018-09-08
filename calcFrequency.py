import nltk
import string
import os
import remove_tags

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from nltk.stem.porter import PorterStemmer
import numpy as np
from collections import OrderedDict
from operator import itemgetter
from itertools import islice
from collections import defaultdict

path = "./classifierSets/trainning"
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

# Aqui para cada arquivo: remove tags, coloca o texto em lower case, 
# elimina pontuacoes, numerais e caracteres que nao unicode.
counter = 0
for dirpath, dirs, files in os.walk(path):
    for f in files:
        fname = os.path.join(dirpath, f)
        counter = counter + 1
        with open(fname) as pearl:
            text = pearl.read()
            text = remove_tags.remove_html_tags(text)
            text = text.translate(None, string.punctuation)
            text = text.translate(None, '0123456789')
            text = unicode(text, errors='replace')
            token_dict[counter] = text.lower()

# Calcula o TF*IDF com base no documento test.txt
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfidf_matrix = tfidf.fit_transform(token_dict.values())

# Esta e uma classe que faz a selecao dos 10% termos mais
# promissores com base em uma f-measure. Fiz uns testes
# mas nao tive muito sucesso.
# selector = SelectPercentile(f_classif, percentile=10)
# selector.fit(tfidf_matrix, range(0, 178))
# selected_tfidf_matrix = selector.transform(tfidf_matrix)
# columns = np.asarray(tfidf.get_feature_names())
# support = np.asarray(selector.get_support())
# columns_with_support = columns[support]
# print support

# Recupera o nome dos tokes utilizados no calculo do TF*IDF
feature_names = tfidf.get_feature_names()

# Abordagem 2, selectiona os n primeiros termos e cria um set.
# best_features_bag = [ ]
# for x in range(649, 650):
#     # # # Recupera o score TF*IDF do documento utilizado no calculo
#     print x
#     doc = x
#     feature_index = tfidf_matrix[doc,:].nonzero()[1]
#     tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
#     # cria um dicionario dos tokens e TF*IDF calculados
#     featureDict = { }
#     for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
#         featureDict[w] = s
#     # Odena o Dicionario por TF*IDF
#     d = OrderedDict(sorted(featureDict.items(), key=itemgetter(1), reverse=True))
#     d = OrderedDict(islice(d.iteritems(),0,50))
#     # coloca os termos em uma lista
#     for key, value in d.iteritems():
#         best_features_bag.append(key)
# print len(set(best_features_bag))
# print set(best_features_bag)

# Abordagem 3, cria um ranking com base na repeticao de um termo entre os documentos.
def select_best_features(tfidf_matrix, feature_names, classBounds):
    best_features_dict = { }
    best_features_dict = defaultdict(lambda: 0, best_features_dict)
    for x in classBounds:
        # # # Recupera o score TF*IDF do documento utilizado no calculo
        doc = x
        feature_index = tfidf_matrix[doc,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])

        # cria um dicionario dos tokens e TF*IDF calculados
        featureDict = { }
        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            featureDict[w] = s

        # Odena o Dicionario por TF*IDF
        d = OrderedDict(sorted(featureDict.items(), key=itemgetter(1), reverse=True))
        d = OrderedDict(islice(d.iteritems(),0,50))

        # Atualiza o dicionario com os n primeiros termos calculados.
        for key, value in d.iteritems():
            best_features_dict[key] = best_features_dict[key] + 1

    best_features_dict = OrderedDict(sorted(best_features_dict.items(), key=itemgetter(1), reverse=True))
    best_features_dict = OrderedDict(islice(best_features_dict.iteritems(),0,50))
    print best_features_dict

# Calcula lista de features por classe.
print "Course"
select_best_features(tfidf_matrix, feature_names, range(0, 649))
print ""

print "Department"
select_best_features(tfidf_matrix, feature_names, range(649, 777))
print ""

print "Faculty"
select_best_features(tfidf_matrix, feature_names, range(777, 1561))
print ""

print "Other"
select_best_features(tfidf_matrix, feature_names, range(1581, 4194))
print ""

print "Project"
select_best_features(tfidf_matrix, feature_names, range(4194, 4545))
print ""

print "Staff"
select_best_features(tfidf_matrix, feature_names, range(4545, 4639))

print "Student"
select_best_features(tfidf_matrix, feature_names, range(4639, 5786))
print ""


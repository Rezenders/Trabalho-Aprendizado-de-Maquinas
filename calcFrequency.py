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

import csv

# Estes valores sao baseados no uso somente
# dos arquivos na pasta classifierSets\trainning
course_trainning_set = range(0, 649)
department_trainning_set = range(649, 777)
faculty_trainning_set = range(777, 1561)
other_trainning_set = range(1561, 4194)
project_trainning_set = range(4194, 4545)
staff_trainning_set = range(4545, 4639)
student_trainning_set = range(4639, 5786)

# Estes valores sao baseados no uso somente
# dos arquivos na pasta classifierSets\test
course_testing_set = range(0, 281)
department_testing_set = range(281, 335)
faculty_testing_set = range(335, 675)
other_testing_set = range(675, 1806)
project_testing_set = range(1806, 1959)
staff_testing_set = range(1959, 2002)
student_testing_set = range(2002, 2496)

# Funcao que faz a quebra do texto em palavras para posterior contagem
# Ela e passada para a biblioteca TfidfVectorizer.
# Notem que ela tambem ja faz o processo de stemming.
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

# Removes html tags, ponctuations, numbers, non-unicode chars and lowers case
# @param file_path html_path
def token_from_file(file_path):
    text = remove_tags.remove_tags_from_file(file_path)
    text = text.translate(None, string.punctuation)
    text = text.translate(None, '0123456789')
    text = unicode(text, errors='replace')
    return text.lower()

# Call token_from_file in all files in trainning_path
# @param trainning_path path containing training set
def token_dict_from_dir(trainning_path):
    counter = 0
    token_dict = {}
    for dirpath, dirs, files in os.walk(trainning_path):
        for f in files:
            fname = os.path.join(dirpath, f)
            token_dict[counter] = token_from_file(fname)
            counter = counter + 1

    return token_dict

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
        d = OrderedDict(islice(d.iteritems(),0,30))

        # Atualiza o dicionario com os n primeiros termos calculados.
        for key, value in d.iteritems():
            best_features_dict[key] = best_features_dict[key] + 1

    best_features_dict = OrderedDict(sorted(best_features_dict.items(), key=itemgetter(1), reverse=True))
    best_features_dict = OrderedDict(islice(best_features_dict.iteritems(),0,50))
    return best_features_dict.keys()

def calculateWordFrequency(features_list, tokens):
    frequencies = []
    for feature in features_list:
        frequency = tokens.count(feature)
        frequencies.append(frequency)
    return frequencies

def resolveTrainningCandidateClass(key):
    if key in course_trainning_set:
        return "course"
    elif key in department_trainning_set:
        return "department"
    elif key in faculty_trainning_set:
        return "faculty"
    elif key in other_trainning_set:
        return "other"
    elif key in project_trainning_set:
        return "project"
    elif key in staff_trainning_set:
        return "staff"
    elif key in student_trainning_set:
        return "student"
    else:
        return "unkown"

def resolveTestingCandidateClass(key):
    if key in course_testing_set:
        return "course"
    elif key in department_testing_set:
        return "department"
    elif key in faculty_testing_set:
        return "faculty"
    elif key in other_testing_set:
        return "other"
    elif key in project_testing_set:
        return "project"
    elif key in staff_testing_set:
        return "staff"
    elif key in student_testing_set:
        return "student"
    else:
        return "unkown"

def generateTrainningSet(features_list):
    trainning_path = "./classifierSets/trainning"
    token_dict = token_dict_from_dir(trainning_path)

    trainning_set = []

    for key, value in token_dict.iteritems():
        tokens = tokenize(value)
        frequencies = calculateWordFrequency(features_list, tokens)
        frequencies.append(resolveTrainningCandidateClass(key))
        trainning_set.append(list(frequencies))

    features_list.append("classification")
    trainning_set.insert(0, features_list)

    return trainning_set

def generateTestingSet(features_list):
    test_path = "./classifierSets/test"
    token_dict = token_dict_from_dir(test_path)

    testing_set = []

    for key, value in token_dict.iteritems():
        tokens = tokenize(value)
        frequencies = calculateWordFrequency(features_list, tokens)
        frequencies.append(resolveTestingCandidateClass(key))
        testing_set.append(list(frequencies))

    features_list.append("classification")
    testing_set.insert(0, features_list)

    return testing_set

def generateAllDataSet(features_list):
    trainning_path = "./classifierSets/trainning"
    token_dict = token_dict_from_dir(trainning_path)


    trainning_set = []

    for key, value in token_dict.iteritems():
        tokens = tokenize(value)
        frequencies = calculateWordFrequency(features_list, tokens)
        frequencies.append(resolveTrainningCandidateClass(key))
        trainning_set.append(list(frequencies))


    test_path = "./classifierSets/test"
    token_dict = token_dict_from_dir(test_path)

    testing_set = []

    for key, value in token_dict.iteritems():
        tokens = tokenize(value)
        frequencies = calculateWordFrequency(features_list, tokens)
        frequencies.append(resolveTestingCandidateClass(key))
        testing_set.append(list(frequencies))

    all_data_set = trainning_set + testing_set
    features_list.append("classification")
    all_data_set.insert(0, features_list)

    return all_data_set

def generateCsv(fileName, data):
    csv_file = open(fileName, "w")
    writer = csv.writer(csv_file, lineterminator='\n')

    for line in data:
        writer.writerow(line)


def main():
    trainning_path = "./classifierSets/trainning"
    token_dict = token_dict_from_dir(trainning_path)

    # Calcula o TF*IDF
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(token_dict.values())

    # Recupera o nome dos tokes utilizados no calculo do TF*IDF
    feature_names = tfidf.get_feature_names()

    # Calcula lista de features por classe.
    print "Course"
    best_features_course = select_best_features(tfidf_matrix, feature_names, course_trainning_set)
    print best_features_course
    print ""

    print "Department"
    best_features_department = select_best_features(tfidf_matrix, feature_names, department_trainning_set)
    print best_features_department
    print ""

    print "Faculty"
    best_features_faculty = select_best_features(tfidf_matrix, feature_names, faculty_trainning_set)
    print best_features_faculty
    print ""

    print "Other"
    best_features_other = select_best_features(tfidf_matrix, feature_names, other_trainning_set)
    print best_features_other
    print ""

    print "Project"
    best_features_project = select_best_features(tfidf_matrix, feature_names, project_trainning_set)
    print best_features_project
    print ""

    print "Staff"
    best_features_staff = select_best_features(tfidf_matrix, feature_names, staff_trainning_set)
    print best_features_staff
    print ""

    print "Student"
    best_features_student = select_best_features(tfidf_matrix, feature_names, student_trainning_set)
    print best_features_student
    print ""

    #Cria um set com todas as features extraidas
    features_set = set()
    features_set.update(best_features_course)
    features_set.update(best_features_department)
    features_set.update(best_features_faculty)
    features_set.update(best_features_other)
    features_set.update(best_features_project)
    features_set.update(best_features_staff)
    features_set.update(best_features_student)

    print "Total de Features: ", len(features_set)

    # #Cria um CSV com os dados para treino
    # trainning_set = generateTrainningSet(list(features_set))
    # generateCsv("trainningSet.csv", trainning_set)
    #
    # #Cria um CSV com os dados de test
    # testing_set = generateTestingSet(list(features_set))
    # generateCsv("testingSet.csv", testing_set)

    all_data_set = generateAllDataSet(list(features_set))
    generateCsv("allData.csv", all_data_set)

if __name__ == "__main__":
    main()

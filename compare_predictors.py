
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
import sklearn

#Compares multiple predictors using the same data
#@param (predictor_name, predictor) list of predictors using the sklearn api
#@param csv_file_location location of the csv file with the data to use
#@param used_features list of the names of the featurs in the csv file
#   that are going to be used for classification
#@param prediction_feature the name of hte feature that is going to be predicted
#@returns dictionary with entries (predictor_name, performance)
def compare_predictors(predictors, csv_file_location, used_features, predicting_feature):

    errors = {}
    #data is a HUGE csv file where each entry is one document
    #and each column correspond to a word considered useful for classification
    data = pd.read_csv(csv_file_location, index_col=False)#, dtype={"class": str })
    #this command was needed in my test dataset. It is not going to be needed
    #in the final code
    #data = data[used_features].dropna(axis=0, how='any')
    predicting_feature_as_int_column = predicting_feature+".value"
    data[predicting_feature_as_int_column], class_labels = pd.factorize(data[predicting_feature], sort=True)

    train_set, test_set = train_test_split(data, test_size=0.3, random_state=int(time.time()))

    for name, predictor in predictors:

        predictor.fit(
            train_set[used_features].values,
            train_set[predicting_feature_as_int_column]
        )

        predictions = predictor.predict(test_set[used_features])

        test_set.loc[:,name + ".prediction"] = predictions

        errors[name] = {}
        #calculate error
        for i in range(len(class_labels)):
            this_class_instances = test_set[test_set[predicting_feature_as_int_column] == i]
            error_qtt = (this_class_instances[predicting_feature_as_int_column] != this_class_instances[name+".prediction"]).sum()
            performance = 1 - (error_qtt / float(this_class_instances.shape[0]))
            errors[name][class_labels[i]] = performance

        overall_performance = float(1) - (test_set[predicting_feature_as_int_column] != predictions).sum()/float(test_set.shape[0])
        errors[name]["overall"] = overall_performance
        #import graphviz
        #dot_data = sklearn.tree.export_graphviz(predictor, out_file=None)
        #graph = graphviz.Source(dot_data)
        #graph.render()

    return errors


########test############

csv_file_location = './allData.csv'
#get the used features from the first line of the csv
#removing the last character because it is \n
used_features = open(csv_file_location).readlines()[0][:-1].split(',')
#the feature we are predicting should be the last one listed
predicting_feature = used_features[-1]
used_features.remove(predicting_feature)

naive_bayes = ("Naive bayes", GaussianNB())
random_forest = ("Random forest", RandomForestRegressor())
dummy = ("Dummy", DummyRegressor())
decision_tree = ("Decision tree", DecisionTreeRegressor())
neural_net = ("Neural net", MLPClassifier(hidden_layer_sizes=(100,50)))

predictors = [naive_bayes, random_forest, decision_tree, neural_net, dummy]

errors = compare_predictors(predictors, csv_file_location, used_features, predicting_feature)
for predictor in errors.keys():
    print("{}:".format(predictor))
    for classification in errors[predictor].keys():
        if classification != "overall":
            print("\t{}: {}%".format(classification, str(errors[predictor][classification]*100)))
    print("\n\t{}: {}%".format("overall", str(errors[predictor]["overall"]*100)))

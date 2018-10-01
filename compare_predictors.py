import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyRegressor
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import sklearn
from sklearn.utils import resample

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
    #data = pd.read_csv(csv_file_location, index_col=True)#, dtype={"class": str })
    #this command was needed in my test dataset. It is not going to be needed
    #in the final code
    #data = data[used_features].dropna(axis=0, how='any')
    predicting_feature_as_int_column = predicting_feature+".value"
    data[predicting_feature_as_int_column], class_labels = pd.factorize(data[predicting_feature], sort=True)

    train_set, test_set = train_test_split(data, test_size=0.2, random_state=int(time.time()))
    test_set = test_set.copy()

    train_set = re_sample(train_set, predicting_feature, 123)

    for name, predictor in predictors:
        #train_set[used_features].values


        #print("name: ", name, " predictor ", predictor)
        predictor.fit(
            train_set[used_features].values,
            train_set[predicting_feature_as_int_column],
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

        if(name == "Decision tree"):
            import graphviz
            dot_data = sklearn.tree.export_graphviz(predictor, out_file=None, feature_names = used_features)
            graph = graphviz.Source(dot_data)
            graph.render()

    return errors

def re_sample(df, predicting_feature, seed):
    #df = pd.read_csv(csv_file_location, index_col=False)
    classes = df[predicting_feature].unique()
    class_qtt = {}
    max_quantity = 0

    for classification in classes:
        quantity = (df[predicting_feature] == classification).sum()
        class_qtt[classification] = quantity
        if quantity > max_quantity:
            max_quantity = quantity

    target = sorted(class_qtt.iteritems(), key=lambda (k,v):(v,k))[6][1]
    for classification in classes:
        this_class = df[df.classification == classification]
        df = df[df.classification != classification]
        replace = target > class_qtt[classification]
        new_sampled_class = resample(this_class, replace=replace, n_samples=target, random_state=seed)
        df = pd.concat([df, new_sampled_class])
    #name, extension = csv_file_location.split('.')
    #csv_file_location = name + "Resampled" + '.' + extension
    #df.to_csv(csv_file_location, index=False)
    return df

########test############
#csv_file_raw = './allData.csv'
#df = pd.read_csv(csv_file_raw, index_col=False)
#dropado = df.drop(['ha','ta'], axis=1)
#dropado.to_csv('./allDataReduced.csv', index=False)

csv_file_location = 'allData.csv'
#get the used features from the first line of the csv
#removing the last character because it is \n
used_features = open(csv_file_location).readlines()[0][:-1].split(',')

#the feature we are predicting should be the last one listed
predicting_feature = used_features[-1]
used_features.remove(predicting_feature)

naive_bayes = ("Naive bayes", GaussianNB())
#random_forest = ("Random forest", RandomForestRegressor())
random_forest = ("Random forest", RandomForestClassifier())
#dummy = ("Dummy", DummyRegressor())
dummy = ("Dummy", DummyClassifier())
#decision_tree = ("Decision tree", DecisionTreeRegressor())
decision_tree = ("Decision tree", DecisionTreeClassifier())
neural_net = ("Neural net", MLPClassifier(hidden_layer_sizes=(100,100,50), max_iter=5000, early_stopping=True))

predictors = [naive_bayes, random_forest, decision_tree, neural_net, dummy]

dataDebug = pd.read_csv(csv_file_location, index_col=False)#, dtype={"class": str })
#dataDebugView = dataDebug.iloc[:,182] # first column of data frame

dataDebugViewLine = dataDebug.iloc[0] # first column of data frame

classes = dataDebug.classification.unique()
#print("Classes: ", classes)

#csv_file_location = re_sample(csv_file_location, predicting_feature, 123)

errors = compare_predictors(predictors, csv_file_location, used_features, predicting_feature)
for predictor in errors.keys():
    print("{}:".format(predictor))
    for classification in errors[predictor].keys():
        if classification != "overall":
            print("\t{}: {}%".format(classification, str(errors[predictor][classification]*100)))
    print("\n\t{}: {}%".format("overall", str(errors[predictor]["overall"]*100)))

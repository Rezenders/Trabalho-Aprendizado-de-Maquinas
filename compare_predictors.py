
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

#Compares multiple predictors using the same data
#@param predictors list of predictors using the sklearn api
#@param csv_file_location location of the csv file with the data to use
#@param used_features list of the names of the featurs in the csv file
#   that are going to be used for classification
#@param prediction_feature the name of hte feature that is going to be predicted
#@returns dictionary with entries (predictor_name, performance)
def compare_predictors(predictors, csv_file_location, used_features, predicting_feature):

    errors = {}

    #data is a HUGE csv file where each entry is one document
    #and each column correspond to a word considered useful for classification
    data = pd.read_csv(csv_file_location, index_col=False)
    #this command was needed in my test dataset. It is not going to be needed
    #in the final code
    data[predicting_feature] = np.where(data[predicting_feature]=="B",0,1)
    data = data[used_features].dropna(axis=0, how='any')

    train_set, test_set = train_test_split(data, test_size=0.3, random_state=int(time.time()))
    #train_set = train_set.astype(int)
    #test_set = test_set.astype(int)

    for predictor in predictors:

        predictor.fit(
            train_set[used_features].values,
            train_set[predicting_feature]
        )

        predictions = predictor.predict(test_set[used_features])

        #calculate error
        performance = float(1) - (test_set[predicting_feature] != predictions).sum()/float(test_set.shape[0])
        errors[predictor] = performance

    return errors


########test############

csv_file_location = './test_data'
used_features = ["ID number", "outcome", "radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave points", "symmetry", "fractal dimension"]
predicting_feature = 'outcome'

naive_bayes = GaussianNB()
random_forest = RandomForestRegressor()
dummy = DummyRegressor()

errors = compare_predictors([dummy, naive_bayes, random_forest], csv_file_location, used_features, predicting_feature)
print(errors)

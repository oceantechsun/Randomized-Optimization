import sklearn as sk  #https://github.com/jorgesleonel/Multilayer-Perceptron/blob/master/Basic%20Multi-Layer%20Perceptron.ipynb
import mlrose as ml
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn import neighbors, datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.utils import shuffle


#url = "/Users/jordan/Desktop/Machine Learning/Randomized Optimization/healthcare-dataset-stroke-data.csv"  
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data"

# Let's start by naming the features
names = ["wife's age", "wife's education","husband's education","number of children","wife's religion","wife employed",\
    "husband's occupation","standard of living","media exposure","contraceptive used"]
"""
names = ["id","malignant","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",\
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",\
        "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",\
            "concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst",\
                "perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst",\
                    "concave points_worst","symmetry_worst","fractal_dimension_worst"]
"""
dataset = pd.read_csv(url, names=names).iloc[1:,:]#.iloc[1:,:]
dataset = dataset[dataset.iloc[:,-1] != 1]
print(dataset)
# Takes first 4 columns and assign them to variable "X"
X = dataset.iloc[:,:-1]
# Takes first 5th columns and assign them to variable "Y". Object dtype refers to strings.
y = dataset.iloc[:, -1].values - 2
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=2)
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

#print(y_train)

#print(y)



ga_nn = ml.NeuralNetwork(hidden_nodes = [2], activation = 'relu', 
                                algorithm = 'genetic_alg', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.1, early_stopping = True, clip_max = 5, 
                                max_attempts = 100, random_state = 2)

rhc_nn = ml.NeuralNetwork(hidden_nodes = [2], activation = 'relu', 
                                algorithm = 'random_hill_climb', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.15, early_stopping = True, clip_max = 10, 
                                max_attempts = 100, random_state = 2)

anneal_nn = ml.NeuralNetwork(hidden_nodes = [2], activation = 'tanh', 
                                algorithm = 'simulated_annealing', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.15, early_stopping = True, clip_max = 10, 
                                max_attempts = 100, random_state = 2)

mimic_nn = ml.NeuralNetwork(hidden_nodes=[2],
                 activation='relu',
                 algorithm='genetic_alg',
                 max_iters=100,
                 bias=True,
                 is_classifier=True,
                 learning_rate=0.1,
                 early_stopping=False,
                 clip_max=5,
                 restarts=0,
                 
                 pop_size=200,
                 mutation_prob=0.1,
                 max_attempts=10,
                 random_state=None,
                 curve=False)


start = time.time()
rhc_nn.fit(X_train, y_train)

in_predictions_rhc = rhc_nn.predict(X_train)
out_predictions_rhc = rhc_nn.predict(X_test)

in_accuracy_rhc = accuracy_score(y_train, in_predictions_rhc)
out_accuracy_rhc = accuracy_score(y_test, out_predictions_rhc)
end = time.time()
total_time_rhc = end-start
confusion_rhc = confusion_matrix(y_test,out_predictions_rhc)
classification_out_rhc = classification_report(y_test, out_predictions_rhc)
classification_in_rhc = classification_report(y_train, in_predictions_rhc)

print(in_accuracy_rhc)
print(out_accuracy_rhc)
print(confusion_rhc)
print(classification_in_rhc)
print(classification_out_rhc)

"""
start = time.time()
anneal_nn.fit(X_train, y_train)

in_predictions_anneal = anneal_nn.predict(X_train)
out_predictions_anneal = anneal_nn.predict(X_test)

in_accuracy_anneal = accuracy_score(y_train, in_predictions_anneal)
out_accuracy_anneal = accuracy_score(y_test, out_predictions_anneal)
end = time.time()
total_time_anneal = end-start
confusion_anneal = confusion_matrix(y_test,out_predictions_anneal)
classification_out_anneal = classification_report(y_test, out_predictions_anneal)
classification_in_anneal = classification_report(y_train, in_predictions_anneal)

print(in_accuracy_anneal)
print(out_accuracy_anneal)
print(confusion_anneal)
print(classification_in_anneal)
print(classification_out_anneal)




start = time.time()
ga_nn.fit(X_train, y_train)


in_predictions_ga = ga_nn.predict(X_train)
#print(in_predictions_ga)
out_predictions_ga = ga_nn.predict(X_test)

in_accuracy_ga = accuracy_score(y_train, in_predictions_ga)
out_accuracy_ga = accuracy_score(y_test, out_predictions_ga)
end = time.time()
total_time_ga = end-start
confusion_ga = confusion_matrix(y_test,out_predictions_ga)
classification_out_ga = classification_report(y_test, out_predictions_ga)
classification_in_ga = classification_report(y_train, in_predictions_ga)

print(in_accuracy_ga)
print(out_accuracy_ga)
print(confusion_ga)
print(classification_in_ga)
print(classification_out_ga)
#print(in_predictions_ga)
#rhc_nn
#anneal_nn
#mimic_nn

"""
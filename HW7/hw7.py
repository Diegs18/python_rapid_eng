###################################################################################################################
#  This program creates a list of lists. The inner lists contains important parameters like test set size, and random
#  state. Next in the list is classifier object with the rest of its parameters that are specific to the obeject. 
#  Lastly in the inner list is a string to be printed out with that classifier so we can tell which is which. The outer
#  List is a list of these list objects described previously. The program takes each list object and creates and runs a
#  the model and prints out the accuracy of the classifier. It does this for each classifier in the list of lists.
#
#  This program continues on from project 1. The predictions are re-ran with a common test set and size to align the 
#  predictions. The program then takes those predictions and votes to get a new prediction. The results of those 
#  Predictions are then printed to the screen. 
#
#   Author: Nicholas DiGregorio 1220871392
###################################################################################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# constants
TEST_PERCENT = 0
RAND_STATE   = 1
CLASS        = 2
NAME         = 3


### Get the data 
df = pd.read_csv('heart1.csv')
x = df.iloc[:, :13]
y = df.iloc[:, 13]
test_amount = np.arange(.2,.4,.01)
scores = np.zeros(len(test_amount))

###################################################################################################
#classifier_list [random state, size of test group, classifier, name of classifier]
###################################################################################################
perceptron = [0.22, 3, Perceptron(max_iter=100, tol=1e-7, eta0=0.0000001, fit_intercept=True, random_state=2, verbose=False, shuffle=True, warm_start=True), "Perceptron:           "] #Full data test acurracy = 83.0% 
logi_reg = [0.27, 5, LogisticRegression(C=10, solver='liblinear', multi_class='ovr', random_state=1, fit_intercept=True, warm_start=True), "Logistic Regression:  "]                   #Full data test acurracy = 85.9%
svm = [0.25, 5, SVC(kernel='rbf', tol= 1e-3, random_state=0, gamma = 0.01, C=100), "Kernal SVM:           "]                                                                           #Full data test acurracy = 92.2%
decision_tree = [0.31, 5, DecisionTreeClassifier(criterion='entropy', splitter='random' ,max_depth=5, random_state=0), "Decision Tree:        "]                                       #Full data test acurracy = 90.7%
random_forest = [0.2, 5, RandomForestClassifier(criterion='gini', n_estimators=13, random_state=0, n_jobs=4), "Random Forest:        "]                                                #Full data test acurracy = 98.9% 
k_near_neighbor = [0.2, 5,KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski'), "K-Nearest Neighbor:   "]                                                                      #Full data test accuracy = 89.3%
classifier_list = [perceptron, logi_reg, svm, decision_tree, random_forest, k_near_neighbor]

###################################################################################################
# Loop through the list of classifiers and print out the results from each
###################################################################################################

for i in range (len(classifier_list)):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = classifier_list[i][TEST_PERCENT], random_state=classifier_list[i][RAND_STATE])

    #fitting the data
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    #create and train the classifier
    classifier_list[i][CLASS].fit(x_train_std, y_train)  #grab the classifier and put it into a variable thats easier to reference. 

    #use the classifier to predict on the test data
    y_pred = classifier_list[i][CLASS].predict(x_test_std)

    score = accuracy_score(y_test, y_pred)


    print('Test accuracy of', classifier_list[i][NAME], '%.5f' % score)
print()


###################################################################################################
# HW7 begins here
###################################################################################################
PCPTN = 0 
LGRN  = 1
SVM   = 2
DT    = 3
RF    = 4
KN    = 5
top_clf = [PCPTN, RF, KN] 
clf_to_add = [LGRN, SVM]
threshes = [4.5, 6, 7.5]
y_pred_list = []


print('Running Predicitons again with a common test set and size so that my predicitons refer to the same values')
#need to redo predictions so I have the same test set since I used different test sets/sizes previously
for i in range (len(classifier_list)):
    y_pred = classifier_list[i][CLASS].predict(x_test_std)
    score = accuracy_score(y_test, y_pred)
    print('Test accuracy of', classifier_list[i][NAME], '%.5f' % score)
    y_pred_list.append(y_pred)
yp_sum = np.zeros_like(y_pred)

print()
print("Starting Ensemble Predictions:")
    
for i in range(len(top_clf)):
    for clf in top_clf: #sum up the predictions
        yp_sum += y_pred_list[clf]
    thresh =  threshes[i] #get the threshold
    if i == 1:  #for 4 methods 
        results = np.where(yp_sum >= thresh, 2, 1) #replace the predictions to 1 and 2 to mimic original predictions
    else:
        results = np.where(yp_sum > thresh, 2, 1) #replace the predictions to 1 and 2 to mimic original predictions
    
    score = accuracy_score(y_test, results)
    if i == 1: #for 4 methods
        print("Number of Classifiers:", 3+i, "The score was:", round(score,5), "Ties were included")
    else:
        print("Number of Classifiers:", 3+i, "The score was:", round(score,5))
    
    try: #only adding 2 classifiers but iterating 3 times 
        top_clf.append(clf_to_add[i])
    except:
        print()
    yp_sum = np.zeros_like(y_pred)


print()


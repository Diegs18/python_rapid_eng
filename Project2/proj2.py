####################################################################################
# This program opens a csv file and takes the data in that file to and trains a
# model to a set of data to classify the data into rocks or mines. The file actually 
# trains on the data twice and then prints out the resluts the second time. It prints
# the test accuracy, and how many components were used to achieve that accuracy as well
# as the best accuracy and number of components to achieve that accuracy. It finally 
# prints the corresponding confusion matrix. 
#
# Author: Nicholas DiGregorio 1220871392
####################################################################################


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from warnings import filterwarnings
from sklearn.metrics import confusion_matrix



####################################################################################
# mat_to_ser: 
#   Transform a matrix to a series that is in order from most related to least related
#   and takes out the "0" entries
# Inputs: 
#   Mat: a square matrix
# Returns: 
#   A sorted Series that lists from max value to min value
####################################################################################
def mat_to_ser(mat):
    mat *= np.tri(*mat.values.shape, k=-1).T
    mat_unstack = mat.unstack()
    mat_unstack_og = mat_unstack.copy()
    mat_unstack = mat_unstack_og.sort_values(ascending=False)
    mat_unstack = mat_unstack[mat_unstack != 0]
    return mat_unstack



####################################################################################
# get_output_correlation:
#   pass in a data frame and print the top correlation variables as well as the 
#   variables that are the top corrlation with the output
# Inputs:
#   df = datafram
# Returns:
#   list of the indexes of variables corrleating with outputs from highest to lowest
####################################################################################
def get_output_correlation(df):
    corr = df.corr().abs()
    corr_unstack = mat_to_ser(corr)

    return corr_unstack[60].index.tolist()

####################################################################################
# classify:
#   Pass in training set and your test set and train your classifier
#   
# Inputs:
#   x_train: training data wrt the inputs
#   x_test : data you are trying to predict on 
#   y_train: values used to train the classifier
#   y_test : the values used to se how you did
# Returns:
#   test_score : the score from the results based on the test data
#   full_score : the score from the results based on the full set of data
####################################################################################

def classify (x_train, x_test, y_train, y_test):
     #fitting the data
        sc = StandardScaler()
        sc.fit(x_train)
        x_train_std = sc.transform(x_train)
        x_test_std = sc.transform(x_test)

        #60,30,20,10,5
        clf =  MLPClassifier(hidden_layer_sizes=(60,30,20,10,5), activation='relu',  max_iter=2000, alpha=0.0001, solver='lbfgs', tol=0.0001, random_state=2) #2
        clf.fit(x_train_std, y_train)

        #use the perecptron to predict on the test data
        #print("Number in test", len(y_test))
        y_pred_test = clf.predict(x_test_std)

        x_comb_std = np.vstack((x_train_std, x_test_std))
        y_comb_std = np.hstack((y_train, y_test))

        y_pred_full = clf.predict(x_comb_std)
        #print('Misclassified samples: %d' % (y_test != y_pred).sum())


        class_labels = [1, 2]
        #confuse = confusion_matrix(y_true=y_comb_std, y_pred=y_pred_full, labels=class_labels)
        confuse = confusion_matrix(y_true=y_test, y_pred=y_pred_test, labels=class_labels)
        test_score = accuracy_score(y_test, y_pred_test)
        full_score = accuracy_score(y_comb_std, y_pred_full)
        
        return test_score, full_score, confuse

##############################################################################################
# Main
##############################################################################################
filterwarnings('ignore')
df = pd.read_csv("sonar_all_data_2.csv",names=np.arange(62))

x = df.iloc[:, :60]
y = df.iloc[:, 60]
df = df.drop(61, axis=1)
fut = [] #feature under test
feat_num = np.arange(0,61)
feat_score = np.zeros(61)

output_corr_i = get_output_correlation(df)
i = 0


for j in range(2): #the model imporves the second time its ran. So running it twice to improve results
    best_tscore = 0
    best_fscore = 0
    best_i = 0
    
    for i in range(60):
        fut.append(output_corr_i[i]) #adding features under test
        col_used = df.iloc[:, fut] #x_columns undertest of are the indexs in the fut
        x_train, x_test, y_train, y_test = train_test_split(col_used, y, test_size = .3, random_state=7) #create the training data
        test_score, full_score, confuse = classify(x_train, x_test, y_train, y_test)  #create the model, fit it, and use it to predict.
        
        if j==1: #dont need to save the data from the first fit of the model
            print(i+1, "Componets got", round(test_score,2), "Accuracy")
            feat_score[i+1] = test_score #save the scores
            
            if(best_fscore < full_score): #save the best scores
                best_fscore = full_score
                best_i = i
                best_tscore = test_score
                best_confuse= confuse
    print("Best Accuracy:", round(best_tscore,2))
    print("This was achieved with", best_i+1, "components")
    
print("The confusion matrix for this case is: ")
print(best_confuse)

plt.plot(feat_num, feat_score)
plt.xlabel("Number of Components")
plt.ylabel("Accuracy")
plt.show()

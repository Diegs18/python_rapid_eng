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


def classify (x_train, x_test, y_train, y_test):
     #fitting the data
        sc = StandardScaler()
        sc.fit(x_train)
        x_train_std = sc.transform(x_train)
        x_test_std = sc.transform(x_test)

        #uncomment the classifier you want to optimize and comment the rest 
        clf = Perceptron(max_iter=100, tol=1e-4, eta0=0.0001, fit_intercept=True, random_state=0, verbose=False, shuffle=True, warm_start=True)
        #clf = LogisticRegression(C=10, solver='liblinear', multi_class='ovr', random_state=1, fit_intercept=True, warm_start=True)
        #clf = SVC(kernel='rbf', tol= 1e-3, random_state=0, gamma = 0.01, C=100)
        #clf = DecisionTreeClassifier(criterion='entropy', splitter='random' ,max_depth=5, random_state=0)
        #clf = RandomForestClassifier(criterion='gini', n_estimators=13, random_state=0, n_jobs=4)
        #clf = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')#neighbors=3 and p=2, 
        clf.fit(x_train_std, y_train) 

        #use the perecptron to predict on the test data
        #print("Number in test", len(y_test))
        y_pred_test = clf.predict(x_test_std)

        x_comb_std = np.vstack((x_train_std, x_test_std))
        y_comb_std = np.hstack((y_train, y_test))

        y_pred_full = clf.predict(x_comb_std)
        #print('Misclassified samples: %d' % (y_test != y_pred).sum())
        test_score = accuracy_score(y_test, y_pred_test)
        full_score = accuracy_score(y_comb_std, y_pred_full)
        
        return test_score, full_score


########################## Design Knobs ##########################
test_group_sizes = np.arange(.2,.4,.01)
rand_nums = np.arange(0,9)

### Get the data 
df = pd.read_csv('heart1.csv')
x = df.iloc[:, :13]
y = df.iloc[:, 13]
test_scores = np.zeros(len(test_group_sizes))
full_scores = np.zeros(len(test_group_sizes))

#print(test_group_sizes)
t_biggest_index = 0
t_biggest_test_score = 0.0
t_biggest_full_score = 0.0
t_biggest_rand  = 0

f_biggest_index = 0
f_biggest_test_score = 0.0
f_biggest_full_score = 0.0
f_biggest_rand  = 0

#classifiers
#perceptron = [5,27, Perceptron(max_iter=100, tol=1e-4, eta0=0.0001, fit_intercept=True, random_state=0, verbose=False, shuffle=True, warm_start=True)]
#logi_reg = LogisticRegression(C=10, solver='liblenear', multiclass='ovr', random_state=0)

for rand in rand_nums: #iterating through different random states
    for j, group_size in enumerate(test_group_sizes): #iterate through group sizes
    #Split into test and training sets: 70% training and 30%
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = group_size, random_state=rand)
        test_scores[j], full_scores[j] = classify(x_train, x_test, y_train, y_test)
       

    for i in range(len(test_scores)):
        #print('Index:', i, 'Test set size:', round(test_group_sizes[i], 5), 'Accuracy: %.5f' % scores[i])
        if test_scores[i] > t_biggest_test_score:
            t_biggest_test_score = test_scores[i]
            t_biggest_full_score = full_scores[i]
            t_biggest_index = i
            t_biggest_rand = rand
        if full_scores[i] > f_biggest_full_score:
            f_biggest_test_score = test_scores[i]
            f_biggest_full_score = full_scores[i]
            f_biggest_index = i
            f_biggest_rand = rand

    #plt.plot(test_group_sizes, scores)
print()
print("Info from best scores from the test perspective")
print("Rand state: ", t_biggest_rand, "And best test score: ")
print('Index:', t_biggest_index, 'Test set size:', round(test_group_sizes[t_biggest_index],5))
print('Accuracy of just the test: %.5f' % t_biggest_test_score)
print('Accuracy of Full data: %.5f' % t_biggest_full_score)

print()
print("Info from best scores from the full set of data perspective")
print("Rand state: ", f_biggest_rand, "And best test score: ")
print('Index:', f_biggest_index, 'Test set size:', round(test_group_sizes[f_biggest_index],5))
print('Accuracy of just the test: %.5f' % f_biggest_test_score)
print('Accuracy of Full data: %.5f' % f_biggest_full_score)
#plt.legend(rand_nums)
#plt.show()


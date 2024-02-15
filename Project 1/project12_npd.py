import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

### Get the data 
df = pd.read_csv('heart1.csv')
x = df.iloc[:, :13]
y = df.iloc[:, 13]
test_amount = np.arange(.2,.4,.01)
scores = np.zeros(len(test_amount))

rand_nums = np.arange(0,10)
print(test_amount)
biggest_index = 0
biggest_score = 0.0
biggest_rand  = 0
for rand in rand_nums:
    for test_group, j in test_amount:
    #Split into test and training sets: 70% training and 30%
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_group, random_state=rand)


        #fitting the data
        sc = StandardScaler()
        sc.fit(x_train)
        x_train_std = sc.transform(x_train)
        x_test_std = sc.transform(x_test)

        #create and train the perceptron
        ppn = Perceptron(max_iter=100, tol=1e-3, eta0=0.0000001, fit_intercept=True, random_state=0, verbose=False, shuffle=True, warm_start=True)
        ppn.fit(x_train_std, y_train) 

        #use the perecptron to predict on the test data
        #print("Number in test", len(y_test))
        y_pred = ppn.predict(x_test_std)
        #print('Misclassified samples: %d' % (y_test != y_pred).sum())
        scores[j] = accuracy_score(y_test, y_pred)

    for i in range(len(scores)):
        #print('Index:', i, 'Test set size:', round(test_amount[i], 5), 'Accuracy: %.5f' % scores[i])
        if scores[i] > biggest_score:
            biggest_score = scores[i]
            biggest_index = i
            biggest_rand = rand
    print()
    print("And the winnder is: ")
    print('Index:', biggest_index, 'Test set size:', round(test_amount[biggest_index],5), 'Accuracy: %.5f' % biggest_score)

    plt.plot(test_amount, scores)
plt.show()


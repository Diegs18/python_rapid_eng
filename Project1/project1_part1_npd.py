###################################################################################################################
#   This program runs and opens up the data for heart disease and prints out the realtionships of covarience and
#   correlation. It prints the strongest overal relationships between all the variables as well as the variables
#   that have the strongest relationships with heart disease. At the end the program will create a pair plot of all
#   the features including our outut heart disease. 
#
#   Author: Nicholas DiGregorio 1220871392
###################################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
# var_to_var: 
#   Print out how each feature is related to the other features in the series.
# Inputs: 
#   Mat_unstack : a series
# Returns: 
#   nothing
####################################################################################
def var_to_var(mat_unstack, size): #didn't end up needing this
    for index in range(size):
        #print(index)
        #print(df.keys()[index])
        feat = df.keys()[index]
        print()
        print("################# ", feat, " #################")
        try:
            mat_unstack[feat]
            if feat != 'a1p2':
                print(mat_unstack[feat])
        except:
            print(feat, "has no relation to other variables")



####################################################################################
# main
####################################################################################



### Get the data 
df = pd.read_csv('heart1.csv')
df_size = df.shape[1] - 1 # -1 for output column
#print(df)


cov = df.cov().abs()
cov_unstack = mat_to_ser(cov)

print()
print()
print("####################################################################################")
print("Variables with Highest Covariation: ")
print("####################################################################################")
print()
corr = df.corr().abs()
corr_unstack = mat_to_ser(corr)
print(cov_unstack.head(20))
print()
print()
print("####################################################################################")
print("Covariences with hear disease: ")
print("####################################################################################")
print(cov_unstack['a1p2'])
print()
print()
print("####################################################################################")
print("Variables with highest correlation: ")
print("####################################################################################")
print()
print(corr_unstack.head(20))
print()
print()

print("####################################################################################")
print("Correlation with heart disease: ")
print("####################################################################################")
print(corr_unstack['a1p2'])


sns.set(style='whitegrid', context='notebook')
sns.pairplot(df, height = 1)
plt.show()




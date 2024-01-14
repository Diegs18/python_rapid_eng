##################################################
# Program to take inputs from the user as a, b, 
# and c and use it to solver the quadratic formula
# y = ax^2 + bx + c
##################################################

import numpy as np


#grab the inputs from the user
a = int(input("Input coefficient a "))
b = int(input("Input coefficient b "))
c = int(input("Input coefficient c "))

#calc the root portion of the numerator
root =  np.emath.sqrt(b**2 - 4*a*c)

#calc the different numerators
num_p = -b + root
num_n = -b - root

#calc the answers
root1 = num_p / (2*a)
root2 = num_n / (2*a)

#decide what to print
if (root1 == root2):
    print("Double root: ", root1)
else:
    print("Root 1: ", root1)
    print("Root 2: ", root2)
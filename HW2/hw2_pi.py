#########################################################################################
# Program to integrate a function defined in the assignement. The goal is to have the 
# program integrate the function and have it find that the answer is pi. After that the 
# Program will print the value of pi defined in numpy to 8 decimal places. Then it will 
# print how close the value we found in the integration is to the value definined in 
# numpy.pi. To see how the function was converted to z from x see hand clac pdf.
#
# Original eq => integrat from 0 to inf(dx/((1+x)*sqrt(x))) = pi
#
# Nicholas DiGregorio, 1220871392
#########################################################################################
import numpy as np                  #getting the array fns
import matplotlib.pyplot as plt                   #getting the ploting fns
from scipy.integrate import quad  #getting the integration fn


#########################################################################################
# Function: integrate y = a^2 + bx + c
#########################################################################################
#def integrate_y (z): #unsimplified
#    a = 1 + (z / (1 - z))
#    b = np.sqrt(z / (1 - z))
#    c = (1 - z)**2
#    return (1 / (a * b)) * (1 / c)

def integrate_y (z): #simplified
    b = np.sqrt(z*(1-z))
    return 1 / (b)


#########################################################################################
# Main
#########################################################################################
#Variable Set up
START = 0 
END   = 1

#integrate
answer, err = quad(integrate_y, START, END)

#display answers
print("Pi is", round(np.pi, 8))
diff = np.pi - answer
print(f"Difference from numpy.pi is: {diff:.15f}")
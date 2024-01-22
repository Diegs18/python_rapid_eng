#########################################################################################
# Program to integrate ax^2 + bx + c, where a, b, and c are defined in the hw sheet. 
# The program will sweep through a range of 0 to 5 in steps of 0.01. Then we will adjust
# The values of a, b, and c will be changed and the function will be integrated again. 
# The answers of both integrations will be plotted
#########################################################################################
import numpy as np                  #getting the array fns
import matplotlib.pyplot as plt                   #getting the ploting fns
from scipy.integrate import quad  #getting the integration fn


#########################################################################################
# Function: integrate y = a^2 + bx + c
#########################################################################################
def integrate_y (x, a, b):
    return np.sin(x**a)+b

#########################################################################################
# Main
#########################################################################################
STEP  = 0.1
START = 0 
END   = 5 # + STEP

xx = np.arange(START, END, STEP)
num_vals = len(xx)
#answers = np.zeros(num_vals,float)

#for i in range(num_vals):
#    x_val = xx[i]
    
answers, err = quad(integrate_y, START, END, args=(2,3))


#plt.plot(xx,answers)
#plt.xlabel("x vals")
#plt.ylabel("y")
#plt.title("Integral of ax^2 + bx + c")
#plt.show()

print("hello world")
                      




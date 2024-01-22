#########################################################################################
# Program to integrate ax^2 + bx + c, where a, b, and c are defined in the hw sheet. 
# The program will sweep through a range of 0 to 5 in steps of 0.01. Then we will adjust
# The values of a, b, and c will be changed and the function will be integrated again. 
# The answers of both integrations will be plotted
#
# Nicholas DiGregorio, 1220871392
#########################################################################################
import numpy as np                  #getting the array fns
import matplotlib.pyplot as plt                   #getting the ploting fns
from scipy.integrate import quad  #getting the integration fn


#########################################################################################
# Function: integrate y = a^2 + bx + c
#########################################################################################
def integrate_y (x, a, b, c):
    return (a * (x**2)) + (b*x) + c

#########################################################################################
# Main
#########################################################################################
STEP  = 0.01
START = 0 
END   = 5

#Variable Set up
xx = np.arange(START, END + STEP, STEP) 
num_vals = len(xx)
answers1 = np.zeros(num_vals,float) #these are one less step due to the nature of calculating 
answers2 = np.zeros(num_vals,float) #the integral is one less value then there are steps


######################################
# Integral
#   This for loop defines the step 
#   and then integrates over that 
#   step.
######################################
for i in range(num_vals):
    x1 = xx[i] #start of the step

    #Save each integration
    answers1[i], err = quad(integrate_y, 0, x1, args=(2,3,4)) 
    answers2[i], err = quad(integrate_y, 0, x1, args=(2,1,1)) 

    #print("x1 =", x1)  #used for debug
    #print("answers:", answers1[i]) #used for debug



######################################
# Plot the results
######################################

#Creating plots
ax1 = plt.subplot()
ax2 = ax1.twinx()
plt.grid()
ax1.set_xlabel("X vals")

#customize axes 1
ax1.plot(xx,answers1, "r-")
ax1.set_ylabel("y = 2x^2 + 3x + 4")
ax1.yaxis.label.set_color('red')

#customize axes 1
ax2.plot(xx,answers2)
ax2.set_ylabel("y = 2x^2 + 1x + 1")
ax2.yaxis.label.set_color('blue')

plt.title("Integral of x")
plt.show()


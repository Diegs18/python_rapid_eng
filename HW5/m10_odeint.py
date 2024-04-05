# simple example using odeint
# author: allee updated by sdm

import numpy as np                            # import packages
import matplotlib.pyplot as plt
from scipy.integrate import odeint            # this is the ODE solver

# solving dy/dt = -2y initial condition y(0)=1

################################################################################
# Function to return the derivative a position with respect to time.           #
# Inputs:                                                                      #
#    ypos - current position                                                   #
#    time - current time                                                       #
# Outputs:                                                                     #
#    returns the derivative                                                    #
################################################################################

def calc_derivative(ypos, time):
    return -2 * ypos

time_vec = np.linspace(0, 4, 40)              # create the time steps

# odeint arguments: function which calculates the derivative
#                   intial value
#                   an array of time steps
#                   optional: args=()
# and it returns an arra of values

yvec = odeint(calc_derivative, 1, time_vec)   # call the solver

plt.plot(time_vec,yvec)                       # and plot the results
plt.xlabel('time')
plt.ylabel('y',rotation=0)
plt.title('odeint')
plt.show()

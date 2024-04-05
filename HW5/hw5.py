##################################################################################################
# This program uses the odeint solver to solve 3 differnetial equations of growing complexity. Note
# the double derivative problem was made based on the example of scipy.integrate.odeint page. Really
# helpful example there. After each equation is solved, the answer is plotted using matplotlib.
#
#
# Author: Nicholas DiGregorio, 1220871392
##################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint            # this is the ODE solver

################################################################################
# Functions to return the derivative an equation with respect to time.           
# Inputs:                                                                      
#    y: variable                                            
#    t: time                                               
# Outputs:                                                                     
#    returns the derivative                                                    
################################################################################
def p1_deriv(x, t):
    return np.cos(t)

def p2_deriv(y, t):
    return -y + (t**2)*np.exp(-2*t) + 10

def p3_deriv(y, t):
    y1, d1 = y #thinking of y as the initial conditions
    ans = [d1, 25*np.cos(t) + 25*np.sin(t) - 4*d1 - 4*y1] #[y', solve for y'']
    return ans

############################       Main       ####################################
time_vec = np.linspace(0,7,700)

########## P1 ############
init_val = 1
yvec = odeint(p1_deriv, init_val, time_vec)

plt.plot(time_vec, yvec)
plt.xlabel('time')
plt.ylabel('y', rotation=0)
plt.title('y\' = Cos(t)')
plt.show()

########## P2 ############
init_val = 0
yvec = odeint(p2_deriv, init_val, time_vec)

plt.plot(time_vec, yvec)
plt.xlabel('time')
plt.ylabel('y', rotation=0)
plt.title('y\' = -y + (t**2)*np.exp(-2*t)+10')
plt.show()


########## P3 ############
init_val = [1, 1] #y0, y'0
y2 = odeint(p3_deriv, init_val, time_vec)

plt.plot(time_vec, y2[:,0], label='y')
plt.plot(time_vec, y2[:,1], label='y\'')
plt.legend()
plt.xlabel('time')
plt.ylabel('y', rotation=0)
plt.title('y\'\'+4y\'+4y = 25cos(t) + 25sin(t)')
plt.show()




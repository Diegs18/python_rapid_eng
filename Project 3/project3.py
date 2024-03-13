import scipy.optimize as optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
####################################################################################################
# This program runs and finds the current through the circuit. It plots the the Diode current vs
# the source voltage and plots the Diode current vs the Diode voltage. For the second part, the 
# program attempts to find the parameters of this diode by optimizing the input parameters. The program
# then plots its model and the actual measurements vs source voltage. 
#
#
# Author Nicholas DiGregorio, 1220871392
###################################################################################################

P1_VDD_STEP = 0.1 
Q =  1.6021766208e-19
KB= 1.380648e-23

################################################################################
# This function finds the currnet in the diode based on the source voltage     #
# Inputs:                                                                      #
#    src_v     - source voltage                                                #
#    r_value   - value of the resistor                                         #
#    ide_value - value of the ideality                                         #
#    phi_value - value of phi                                                  #
#    area      - area of the diode                                             #
#    temp      - temperature                                                   #
#    meas_i    - measured current                                              #
# Outputs:                                                                     #
#    err_array - array of error measurements                                   #
################################################################################

def find_i_v(src_v,r_value,ide_value, temp, is_value):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P1_VDD_STEP                 # an initial guess for the voltage
    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(est_v, ide_value, temp, is_value)
    return diode_i, est_v

##############################################################################
# This function is the same but calculates the is value for the user
#############################################################################

def find_i_v2(src_v,r_value,ide_value, temp, phi_value, area):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P1_VDD_STEP                 # an initial guess for the voltage
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( KB * temp ) )
    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(est_v, ide_value, temp, is_value)
    return diode_i, est_v


################################################################################
# This function does the optimization for the resistor                         #
# Inputs:                                                                      #
#    r_value   - value of the resistor                                         #
#    ide_value - value of the ideality                                         #
#    phi_value - value of phi                                                  #
#    area      - area of the diode                                             #
#    temp      - temperature                                                   #
#    src_v     - source voltage                                                #
#    meas_i    - measured current                                              #
# Outputs:                                                                     #
#    err_array - array of error measurements                                   #
################################################################################

def opt_r(r_value,ide_value,phi_value,area,temp,src_v,meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P1_VDD_STEP                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( KB * temp ) )
    #is_value = 1e-9
    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(est_v, ide_value, temp, is_value)
    return meas_i - diode_i

################################################################################
# This function does the optimization for the ideality                         #
# Inputs:                                                                      #
#    r_value   - value of the resistor                                         #
#    ide_value - value of the ideality                                         #
#    phi_value - value of phi                                                  #
#    area      - area of the diode                                             #
#    temp      - temperature                                                   #
#    src_v     - source voltage                                                #
#    meas_i    - measured current                                              #
# Outputs:                                                                     #
#    err_array - array of error measurements                                   #
################################################################################

def opt_ide(ide_value, r_value,phi_value,area,temp,src_v,meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P1_VDD_STEP                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( KB * temp ) )

    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)
    return (meas_i - diode_i)/(meas_i + diode_i + 1e-15)

################################################################################
# This function does the optimization for the ideality                         #
# Inputs:                                                                      #
#    r_value   - value of the resistor                                         #
#    ide_value - value of the ideality                                         #
#    phi_value - value of phi                                                  #
#    area      - area of the diode                                             #
#    temp      - temperature                                                   #
#    src_v     - source voltage                                                #
#    meas_i    - measured current                                              #
# Outputs:                                                                     #
#    err_array - array of error measurements                                   #
################################################################################

def opt_phi(phi_value,r_value,ide_value,area,temp,src_v,meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P1_VDD_STEP                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( KB * temp ) )

    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,
				                (src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)
    res = (meas_i - diode_i)/(meas_i + diode_i + 1e-15)
    return res 

################################################################################
# solve_diode_v
#   Inputs:
#       vd:   Diode voltage
#       vs:   Source Voltage
#       r:    Resistance 
#       n:    ideality value    
#       temp: temperature
#       Is:   bias current
################################################################################

def solve_diode_v(est_v, src_v, r_value, ide_value, temp, is_value):
    term1 = (est_v-src_v)/r_value
    term2 = compute_diode_current(est_v, ide_value, temp, is_value)
    return term1 + term2

################################################################################
# compute diode current
#   Inputs:
#       vd:   Diode voltage
#       vs:   Source Voltage
#       r:    Resistance 
#       n:    ideality value    
#       temp: temperature
#       Is:   bias current
################################################################################
def compute_diode_current(est_v, ide_value, temp, is_value):
    vt = ( ide_value * KB * temp ) / Q
    return is_value*( np.exp( est_v / vt ) - 1)


##################################################################################
# Main
##################################################################################


### Read File in 
names_list = ["Vs", "mId"]
df = pd.read_csv("DiodeIV.txt", names=names_list, sep=" ")
#print(df)

#print()
#print(df.head(2))
#print()
#print(df['Vs'])
#print()
#print(df['Vd'])

src_v = df['Vs'].to_numpy()
meas_diode_i = df['mId'].to_numpy()
#print(src_v[0])
################################# Part 1 #########################################

r_val = 11000
ide_val = 1.7
is_val = 1e-9
temp = 350

diode_i, diode_v = find_i_v(src_v,r_val,ide_val,temp, is_val)
#print(diode_i)
ax1 = plt.subplot()

#plotting source V vs diode I
ax1.set_xlabel("Source Voltage (V)")
ax1.set_ylabel("Diode current (log(A)) vs Source Voltage")
ax1.yaxis.label.set_color('red')
ax1.plot(src_v,np.log10(diode_i), 'r-')

#plotting diode V vs doide I
ax2 = ax1.twinx()
ax2.set_xlabel("Diode Voltage (V)")
ax2.set_ylabel("Diode current (log(A)) vs Diode Voltage")
ax2.yaxis.label.set_color('blue')
ax2.plot(diode_v,np.log10(diode_i), 'b-')

plt.title("Diode Current")
plt.grid()
plt.show()


################################# Part 2 #########################################

r_val = 10000
ide_val = 1.5
phi_val = 0.8
P2_AREA = 1e-8
P2_T    = 375
i=0
print("Initial values")
print("Resistor Value:", r_val)
print("Phi Value", phi_val)
print("Ideality Value:", ide_val)
I_LIM=100
error=100
while(error>1e-8 and i<I_LIM):
    print('Iteration:', i)    
    r_val_opt = optimize.leastsq(opt_r,r_val,
                                    args=(ide_val,phi_val,P2_AREA,P2_T,
                                        src_v,meas_diode_i))
    r_val = r_val_opt[0][0]
    print("Resistor Value:", r_val)
   
    phi_val_opt = optimize.leastsq(opt_phi, phi_val, 
                                    args=(r_val,ide_val,P2_AREA,P2_T,
                                          src_v,meas_diode_i))
    phi_val = phi_val_opt[0][0]
    print("Phi Value", phi_val)

    ide_val_opt = optimize.leastsq(opt_ide, ide_val, 
                                    args=(r_val,phi_val,P2_AREA,P2_T,
                                            src_v,meas_diode_i))
    ide_val = ide_val_opt[0][0]
    print("Ideality Value:", ide_val)
    res = opt_phi(phi_val, r_val, ide_val, P2_AREA, P2_T, src_v, meas_diode_i)
    error = np.sum(np.abs(res))/len(res)
    print("Error:", error)
    print()
    i += 1

ax1 = plt.subplot()
#plot the measured diode current vs source voltage
ax1.set_xlabel("Source Voltage (V)")
ax1.set_ylabel("Measured Diode current (log(A)) vs Source Voltage")
ax1.yaxis.label.set_color('red')
ax1.plot(src_v,np.log10(meas_diode_i), 'rx-', markersize=9)

#plot the modeled diode current vs source voltage
model_i, model_vd = find_i_v2(src_v, r_val, ide_val, P2_T, phi_val, P2_AREA)
ax2 = ax1.twinx()
ax2.set_ylabel("Model Diode current (log(A)) vs Diode Voltage")
ax2.yaxis.label.set_color('blue')
ax2.plot(src_v, np.log10(model_i), 'bo-', markersize=5)

plt.title("Diode Current")
plt.grid()
plt.show()

#def opt_r(r_value,ide_value,phi_value,area,temp,src_v,meas_i):
#def opt_ide(ide_value,r_value,phi_value,area,temp,src_v,meas_i):
#def opt_phi(phi_value,r_value,ide_value,area,temp,src_v,meas_i):
#def find_i_v2(src_v,r_value,ide_value, temp, phi_value, area):
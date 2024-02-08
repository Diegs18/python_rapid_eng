################################################################################
# Created on Fri Aug 24 13:36:53 2018                                          #
#                                                                              #
# @author: olhartin@asu.edu; updates by sdm                                    #
#                                                                              #
# Program to solve resister network with voltage and/or current sources        #
# 
# Updated to complete homework assignment on 1/28/2024
# Updated by Nicholas DiGregorio 1220871392
################################################################################

import numpy as np                     # needed for arrays
from numpy.linalg import solve         # needed for matrices
from read_netlist import read_netlist  # supplied function to read the netlist
import comp_constants as COMP          # needed for the common constants

# this is the list structure that we'll use to hold components:
# [ Type, Name, i, j, Value ]

################################################################################
# How large a matrix is needed for netlist? This could have been calculated    #
# at the same time as the netlist was read in but we'll do it here             #
# Input:                                                                       #
#   netlist: list of component lists                                           #
# Outputs:                                                                     #
#   node_cnt: number of nodes in the netlist                                   #
#   volt_cnt: number of voltage sources in the netlist                         #
################################################################################

def get_dimensions(netlist):           # pass in the netlist

    ### EXTRA STUFF HERE!
    node_cnt = 0
    volt_cnt = 0
    
    for comp in netlist:
        if (comp[COMP.TYPE] == 1):
            volt_cnt += 1
        i = comp[COMP.I]
        j = comp[COMP.J]
        if (i > node_cnt):
            node_cnt = i
        if (j > node_cnt):
            node_cnt = j

    print(' Nodes ', node_cnt, ' Voltage sources ', volt_cnt)
    return node_cnt,volt_cnt

################################################################################
# Function to stamp the components into the netlist                            #
# Input:                                                                       #
#   y_add:    the admittance matrix                                            #
#   netlist:  list of component lists                                          #
#   currents: the matrix of currents                                           #
#   node_cnt: the number of nodes in the netlist                               #
# Outputs:                                                                     #
#   node_cnt: the number of rows in the admittance matrix                      #
#   y_add:    the stamped admittance matrix                                    #
#   currents: the stamped current matrix                                       #
################################################################################

def stamper(y_add,netlist,currents,node_cnt):
    # return the total number of rows in the matrix for
    # error checking purposes
    # add 1 for each voltage source...
    
    src_cnt = vlt_cnt #keep track of how many voltage sources we have added to the matrix
    for comp in netlist:                  # for each component...
        #print(' comp ', comp)            # which one are we handling...

        # extract the i,j and fill in the matrix...
        # subtract 1 since node 0 is GND and it isn't included in the matrix
        i = comp[COMP.I] - 1
        j = comp[COMP.J] - 1
        
        ###########################################
        #resistor
        ###########################################
        if (comp[COMP.TYPE] == COMP.R):           # a resistor
            if (i >= 0):                            # add on the diagonal, and we are not on the ground node
                y_add[i][i] += (1 / comp[COMP.VAL])
                
                #for values that requrire both i and j need to have checked for both
                if(j>=0):
                    y_add[i][j] -= (1 / comp[COMP.VAL])
                    y_add[j][i] -= (1 / comp[COMP.VAL])

            #for the one that only requires j need to check seperately otherwise you could miss putting j,j in
            if (j >= 0):
                y_add[j][j] += (1 / comp[COMP.VAL])

        net_size = node_cnt + vlt_cnt
        
        ###########################################
        #voltage source
        ###########################################
        if ( comp[COMP.TYPE] == COMP.VS ): 
            m = net_size - src_cnt
            if(i>=0):
                y_add[m][i] = 1
                y_add[i][m] = 1
            if(j>=0):
                y_add[m][j] = -1
                y_add[j][m] = -1

            currents[m] = comp[COMP.VAL]
            src_cnt -= 1
            #print(y_add)
            #print(currents)

        ###########################################
        #current source
        ###########################################
        if ( comp[COMP.TYPE] == COMP.IS ):           # a current source
            if(i>=0):
                currents[i] -= comp[COMP.VAL]
            if(j>=0):
                currents[j] += comp[COMP.VAL]
            #print(currents)
        
        
        
    print ("Addmitance:\n", y_add)
    print ("Currents:\n",currents)

    return node_cnt # should be same as number of rows!

################################################################################
# Start the main program now...                                                #
################################################################################

# Read the netlist!
netlist = read_netlist()

# Print the netlist so we can verify we've read it correctly
for index in range(len(netlist)):
    print(netlist[index])
print("\n")

#get info for the matricies 
node_cnt, vlt_cnt = get_dimensions(netlist)
net_size = node_cnt + vlt_cnt

#create the matricies with values initialized to zero
y_add = np.zeros((net_size, net_size))
currents =  np.zeros((net_size))
volts =  np.zeros((net_size))

#set up admittance and current matricies
node_cnt = stamper(y_add, netlist, currents, node_cnt)

#get the answer and print it
volts = solve(y_add, currents)
print("\nVoltage Matrix:")
print(volts)

#print("individually formatted")
#for ans in volts: 
#    print(f"{ans:.5f}")





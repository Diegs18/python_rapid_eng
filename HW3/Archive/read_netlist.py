################################################################################
# read netlist file and create a list of lists...                              #
################################################################################

import comp_constants as COMP    # get the constants needed for lists
from sys import exit             # needed to exit on error
#neimport os
################################################################################
# Read a netlist from a spice-like text file                                   #
# Input:                                                                       #
#   none                                                                       #
# Outputs:                                                                     #
#   netlist: a list of components, each component as a list                    #
#                                                                              #
# this is the list structure that we'll use to hold components:                #
# [ Type, Name, i, j, Value ]                                                  #
################################################################################

def read_netlist():              # read a netlist - no input argument!
    #cwd = os.getcwd()  # Get the current working directory (cwd)
    #files = os.listdir(cwd)  # Get all the files in that directory
    #print("Files in %r: %s" % (cwd, files))


    filename=input("enter netlist text file name: ")      # ask for the netlist
    #print(filename)                                      # debug statement
    fh = open(filename,"r")                               # open the file
    lines = fh.readlines()                                # read the file
    fh.close()                                            # close the file

    netlist = []                                          # initialize our list
    for line in lines:                                    # for each component
        line=line.strip()                                 # strip CR/LF
        if line:                                          # skip empty lines

            # reads: name, from, to, value
            # so we need to insert the node type at the start of the list
            # parse properties delimited by spaces
            props = line.split(" ")

            if ( props[COMP.TYPE][0] == COMP.RESIS ):     # is it a resistor?, look at first char of the first thing in the list
                props.insert(COMP.TYPE,COMP.R)            # insert type
                props[COMP.I]   = int(props[COMP.I])      # convert from string
                props[COMP.J]   = int(props[COMP.J])      # convert from string
                props[COMP.VAL] = float(props[COMP.VAL])  # convert from string
                netlist.append(props)                     # add to our netlist

            elif ( props[COMP.TYPE][0:2] == COMP.V_SRC ): # a voltage source?
                props.insert(COMP.TYPE,COMP.VS)           # insert type
                props[COMP.I]   = int(props[COMP.I])      # convert from string
                props[COMP.J]   = int(props[COMP.J])      # convert from string
                props[COMP.VAL] = float(props[COMP.VAL])  # convert from string
                netlist.append(props)                     # add to our netlist

            elif ( props[COMP.TYPE][0:2] == COMP.I_SRC ): # a current source?
                props.insert(COMP.TYPE,COMP.IS)           # insert type
                props[COMP.I]   = int(props[COMP.I])      # convert from string
                props[COMP.J]   = int(props[COMP.J])      # convert from string
                props[COMP.VAL] = float(props[COMP.VAL])  # convert from string
                netlist.append(props)                     # add to our netlist

            else:                                         # unknown component!
                print("Unknown component type:\n",line)   # bad data!
                exit()                                    # bail!

    return netlist

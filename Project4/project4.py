################################################################################
# Project 4                                                                    #
# This program copies a file called invchainbase into invchainrun and then opens 
# invchainrun and begins appening the lines of HSPICE code needed to iterate and
# run HSPICE. The python script loops through many combonations. This program is
# based on the code given by the file run_hspice.py. 
#
# Author: Nicholas DiGreogorio 1220871392
################################################################################

import numpy as np      # package needed to read the results file
import subprocess       # package needed to lauch hspice
import shutil
import matplotlib.pyplot as plt
################################################################################
# Start the main program here.                                                 #
################################################################################



# Create a base point file that can be used through the iterations
#og_file = open('InvChainBase.sp', 'r')
#cp_proc = subprocess.Popen(["cp", "InvChainBase.sp InvChainNPD.sp"],
#                    stdout=subprocess.PIPE,stderr=subprocess.PIPE)
#output, err = cp_proc.communicate()

letters = np.arange(97, 112) #create an array of ascii numbers

best_inv = 0
best_fan = 0
best_tphl = 100
fans = np.arange(2, 7)
for inv_rng in range(1, 11, 2):   
    for fan in fans:
        file_str = "InvChainRun" + ".sp" #+ str(inv_rng)
        shutil.copy("InvChainBase.sp", file_str) #'InvChainRun.sp'
        file = open(file_str, 'a') #'InvChainRun.sp'
        fan_str = ".param fan = " + str(fan) +'\n\n'
        file.write(fan_str) #write the fan parameter
        #code that loops through number of inverters to build the file
        for inv_num in range(1, inv_rng + 1):
            if inv_rng == 1: 
                file.write('Xinv1 a z inv M=1 \n')
            elif inv_rng > 1:
                if inv_num == 1: 
                    inv = 'Xinv'+ str(inv_num) + " " + chr(letters[inv_num-1]) + " " + chr(letters[inv_num]) + " " + 'inv M=1\n'
                    file.write(inv)
                elif inv_num < inv_rng:
                    #ex    xinv   1                     a                              b                              inv M=fan**   1
                    inv = 'Xinv'+ str(inv_num) + " " + chr(letters[inv_num-1]) + " " + chr(letters[inv_num]) + " " + 'inv M=fan**' + str(inv_num-1) +'\n'
                    file.write(inv)
                else: #do the last inverter in the chain
                    inv = 'Xinv'+ str(inv_num) + " " + chr(letters[inv_num-1]) + " z " + 'inv M=fan**' + str(inv_num-1) +'\n'
                    file.write(inv)

        file.write("\n.end\n")
        #og_file.close()
        file.close()
        # launch hspice. Note that both stdout and stderr are captured so
        # they do NOT go to the terminal!
        proc = subprocess.Popen(["hspice","InvChainRun.sp"],
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        output, err = proc.communicate()# extract tphl from the output file
        
        data = np.recfromcsv("InvChainRun.mt0.csv",comments="$",skip_header=3)
        tphl = data["tphl_inv"]
        #check to see if the best prop delay was achieved
        if(tphl < best_tphl ): 
            best_tphl = tphl
            best_fan = fan
            best_inv = inv_rng
        #print the run to the screen
        print('N', inv_num, 'fan', fan, 'tphl', tphl)
#print the best one found
print("Best Values were")
print("fan =", best_fan)
print("num of inverters =", best_inv)
print("tphl =", best_tphl)

  


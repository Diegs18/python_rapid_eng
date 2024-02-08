import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

###################################################################################################
#  This script creates and launces a GUI that calculates the wealth over the course of 70 years
#  based on the parameters provided by the user at runtime. This script will then plot the 
#  cacluclated wealth and displays 10 lines representing 10 runs of 70 years.
#  A variable called debug can be set that bypasses the gui and sends hardcoded variables into the
#  function for quicker action of calc_rtns. These hard coded vars can be found in the main part 
#  of the program.
#
#  Author: Nicholas DiGregorio 1220871392
###################################################################################################


###################################################################################################
# Constants
#
###################################################################################################

RATE         = 0
STD_DEV      = 1
YEARLY_CONT  = 2
NUMY_CONT    = 3
NUMY_TO_RET  = 4
ANNUAL_SPEND = 5
MAX_YEARS    = 70

FIELD_NAMES = ["Mean Return", "Standard Deviation Return", "Yearly Contribution", "Number of Years of Contribution", "Number of Years to Retirement", "Annual Spend in Retirement"]
FIELD_NUM = 6

###################################################################################################
# Calc_rtns: 
#   Calculate the returns of the hypothetical scenario provided by the user. This function plots each
#   run of 70 years with a total of 10 runs. If the deb_flg is set, the entries are coming from
#   hardcoded values rather than the gui. This creates a couple special case scenarios where there is
#   no label created and how we get the entries out of the array. This function handles that based
#   the value of dbg_flg.
#
# Inputs:
#   entries: the variables passed in by the user. In the order of the FIELD_Names List constant
#   lab:     the label object that is passed in from the gui. If debuging set this to 0
#   dbg_flg: a flag used to make decisions that allow the programmer to send in hardcoded values 
#            and to still be able to run the script without errors
# Outputs: 
#   Nothing
###################################################################################################
def calc_rtns(entries, lab, dbg_flg):
    #variable init
    e = np.zeros(len(entries))
    w = np.zeros(MAX_YEARS+1)
    x = np.arange(MAX_YEARS+1)
    years = 0
    brk_flg = 0  
    ret_flg = 0
    ret_sum = 0

    #set up entries, changing from entries to e because e is easier to use but less descriptive
    if(not dbg_flg): #called by the gui
        for i in range(len(entries)):
            e[i] =  float(entries[i].get()) #need to cast them
            #print(FIELD_NAMES[i], entries[i].get())
    else: #called for debugging purposes
        for i in range(len(entries)):
            print("Debug", FIELD_NAMES[i], entries[i])
        e = entries

    #we have all the input data, calculate and plot the data
    for i in range(10):
        noise = (e[STD_DEV] / 100) * np.random.randn(MAX_YEARS)
        for year in range(len(w)):
            if(w[year]<=0 and year > 0): #want to quite when the wealth hits zero or less but dont want to quit on the first year
                w[year] = 0
                brk_flg = 1 #remember if we quit the loop early
                years = year
                break; 
            if(year <= e[NUMY_CONT]):
                w[year+1] = w[year]*(1 + e[RATE]/100 + noise[year]) + e[YEARLY_CONT]
            elif(year <= e[NUMY_TO_RET]):
                w[year+1] = w[year]*(1 + e[RATE]/100 + noise[year])
            elif(year < MAX_YEARS):
                if (ret_flg == 0):
                    ret_flg = 1
                    ret_sum += w[year]
                    #print("We are this wealthy this time:\t{:,}" .format(w[year]))
                w[year+1] = w[year]*(1 + e[RATE]/100 + noise[year]) - e[ANNUAL_SPEND]
            #print("Year:", year, "Wealth:",w[year]) #debugging
        if(brk_flg): #did we leave the loop early?
            x2 = np.arange(years+1)
            w2 = np.zeros(years+1)
            for i in range(years + 1): 
                w2[i] = w[i]
            brk_flg = 0
            #print("W2 is:", w2) #Debugging
            plt.plot(x2, w2, marker = 'x')
        else: #we finished the loop normally
            plt.plot(x,w, marker = 'x')
        w.fill(0)
        ret_flg = 0
    ave_wealth = round(ret_sum/10)    
    if(not dbg_flg):
        #print("Average wealth at ret\t{:,}" .format( ret_sum/10))
        lab.configure(text = "Average Wealth at Retirment: \t{:,}" . format(ave_wealth))

    plt.xlabel("Years")
    plt.ylabel("Wealth")
    plt.title("Wealth over 70 Years")
    plt.show()   
    
           



###################################################################################################
# makeforme:
#   This function takes in the root widget and and packs int the labels, entries, and buttons that
#   that are required for the GUI in the homework
#
# Inputs: 
#   root widget
#
# Outputs: 
#   Nothihng
###################################################################################################
def makeform(root):
    entries = [] #store the values of each entry into this list, this gets passed into the calc_rtn call
    for index in range(FIELD_NUM): #create the side by side label and entries 
        row = tk.Frame(root, bd = 5, width=200, height = 20)
        lab = tk.Label(row, width=30, text=FIELD_NAMES[index]+": ", anchor='w') #create a label object 
        ent = tk.Entry(row) #create an entry object
        ent.insert(0, "0")

        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append(ent) #add the entry object to the list
    
    #do the label
    row = tk.Frame(root, bd = 5, width=200, height = 20)
    lab = tk.Label(row, text ="Average Wealth at Retirment") #this gets passed into the calc_rtns call
    row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    lab.pack(side=tk.LEFT)
    
    #do the buttons
    row = tk.Frame(root, bd = 5, width=200, height = 20)
    row.pack(side=tk.TOP, padx=5, pady=5)
    b1 = tk.Button(root, text="Quit", command = (root.destroy))
    b1.pack(side=tk.LEFT, padx=5, pady=5)

    b1 = tk.Button(root, text="Calculate Returns", command = (lambda e=entries : calc_rtns(e, lab, 0))) #when calling calc pass zero in for not debug action
    b1.pack(side=tk.RIGHT, fill=tk.X, padx=5, pady=5)


 ###################################################################################################
# Main Loop
#
###################################################################################################
dbg = 0 #To run the gui set dbg = 0. Otherwise call calc_rtns without using the gui to make life easier
if(dbg): 
    entries = [0, 0, 0, 0, 0, 0,]
    entries[RATE        ] = 8
    entries[STD_DEV     ] = 20
    entries[YEARLY_CONT ] = 10000
    entries[NUMY_CONT   ] = 30
    entries[NUMY_TO_RET ] = 20
    entries[ANNUAL_SPEND] = 80000

    average_wealth = calc_rtns(entries, 0, 1) 
else: 
    root = tk.Tk()
    makeform(root)
    root.mainloop()
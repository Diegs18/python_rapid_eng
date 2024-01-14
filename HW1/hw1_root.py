#############################################################################################
# This problem is split into 2 parts the first is to include the recursion function from the 
# book. Run it, and understand it. The second part is to do this recursivivly calculate the
# square root of a number using this method defined in the problem. Need to keep calling the 
# function until the number calculated is within a tolerance of +/- .01
#############################################################################################
import numpy as np


###############################################
# Factorial example from the book
###############################################
# def factorial (n): 
#     if n==1: 
#         return 1
#     else:
#         return n*factorial(n-1)
# 
# num = 3
# print("factorial of", num, "is:", factorial(num))


###############################################
# square root calculation
###############################################
import numpy as np
#define a function to do the calculation and return the result
def calc(number, guess):
    num = guess + number/guess
    return num/2 


#define a function to do the recursion
def root_fn(number, guess):
    next_guess = calc(number, guess) #do the calculation
    if(abs(guess - next_guess)<0.01):
        return next_guess #we found it!
    else: 
        return root_fn(number, next_guess) #keep guessing

#get input from the user
input_num = int(input("Enter a number whose square root is desired: "))
input_guess = int(input("Enter an initial guess: "))

root = root_fn(input_num, input_guess)
print("The square root of", input_num, "is", round(root, 2))
#print("The numpy square root is", np.emath.sqrt(input_num)) #uncomment for comparison

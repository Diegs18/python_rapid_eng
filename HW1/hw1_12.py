#################################################################
# This is part 2 of hw1. The goal is to print a list of of prime 
# numbers between 2 and 10000. The helpful hints from the problem
# are listed below:
#   a. Only need to check if it is divisible by other primes
#   b. Only check for factors up to and including sqrt(n)
#   c. If there is a single prime factor, no need to keep looking
#      number is not prime
#################################################################

import numpy as np
MAX_VAL = 10000 #debug with smaller lists is easier
prime_list = [2]
is_prime = True
#n is the number between 3 and 10000 that we are finding primes for
for n in range(3, MAX_VAL+1):
    rootn = np.emath.sqrt(n)
    #r is the factor that we are checking in n
    for r in prime_list:
        if (n % r == 0): 
            #n is not prime set the flag and break
            is_prime = False
            break; #once we found a factor by definition it is not prime
        #else: try again 
        if (r>rootn): #need to include rootn, so do one more to make sure root n is included
            break; #if the factor we are checking is greater then the root of the number its not prime by math tricks
    #if we found a prime number add it to the list
    if (is_prime == True):
        prime_list.append(n)
    else: 
        is_prime = True; #need to reset the is_prime flag



print(prime_list)
#print("The length of the prime list is: ", len(prime_list)) #uncomment this for debugging
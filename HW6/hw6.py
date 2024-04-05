##############################################################################################
# This program runs and uses pythagorean theorm on a pair of random numbers to see if the
# hypotenus is less than 1. If it is less than 1, it is counted. If its greater or equal to one
# it is not counted. The count is then divided by the total number of iterations and multiplied
# by 4. If this is within a precision of pi we count it as a success. We average the successful
# pi values at each precision and print it out at the of the loop.
#
#
# Author: Nicholas DiGregorio, 1220871392
##############################################################################################

import numpy as np
from random import random

precisions = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
for prec in precisions:
    success_cnt = 0
    sum_pi = 0
    for j in range(1,101):
        pi_cnt = 0
        ave_pi = 0
        for i in range(1, 10001):
            x = random()
            y = random()
            z = np.sqrt(x**2 + y**2)
            if z<1:
                pi_cnt += 1
            calc_pi = 4*pi_cnt/i
            if abs(np.pi - calc_pi) < prec:
                sum_pi += calc_pi
                success_cnt += 1
                break;
    #ave_pi = sum_pi

    if(success_cnt<1):
        print(prec, "no success")
    else:    
        ave_pi = sum_pi/success_cnt
        print(prec, "success", success_cnt, "times", ave_pi)
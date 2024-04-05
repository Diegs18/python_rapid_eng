##############################################################
#
#
#
##############################################################

import numpy as np
from random import random

precisions = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
for prec in precisions:
    for j in range(1,101):
        pi_cnt = 0
        sum_pi = 0
        ave_pi = 0
        success_cnt = 0
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
                if success_cnt == 100:
                    break

    if(success_cnt<1):
        print(prec, "no success")
    else:    
        ave_pi = sum_pi/success_cnt
        print(prec, "success", success_cnt, "times", ave_pi)
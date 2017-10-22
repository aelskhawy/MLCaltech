# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 02:26:55 2017

@author: Mrskhawy
"""

import numpy as np 
import math 

N=460000
#delta= 4*math.pow(2*N, 10)*np.exp(-(0.05*0.05)*N/8) 

Omega=math.sqrt( (8/N)* np.log( (4* math.pow(2*N,10))/0.05) )

print(Omega)

print("smallest N for which Omega <= 0.05 is ", N)
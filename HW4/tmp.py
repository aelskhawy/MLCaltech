# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 03:11:06 2017

@author: Mrskhawy
"""

import numpy as np 
import matplotlib.pyplot as plt
import math 
from numpy.linalg import inv

data = np.random.uniform(-1., 1., size=(2,1)) # inputs X1, X2 
output_TF =np.sin(math.pi * data )  # output of the target function 
X_pseudo= inv((data.T).dot(data)).dot(data.T)
a=X_pseudo.dot(output_TF)

plt.scatter(data, output_TF)
line_x1 = np.linspace(-1, 1, 10)
line_x2 = a[0][0] * line_x1 
plt.plot(line_x1, line_x2)


data = np.random.uniform(-1., 1., size=(2,2)) # inputs X1, X2 
data_set =  np.column_stack((np.ones(2), data)) # [1,x1,x2]    

output_TF =np.sin(math.pi * data )  # output of the target function 
X_pseudo= inv((data.T).dot(data)).dot(data.T)
a=X_pseudo.dot(output_TF)

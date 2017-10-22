# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 23:50:29 2017

@author: Mrskhawy
"""

import numpy as np 
import matplotlib.pyplot as plt
import math 
from numpy.linalg import inv

x=np.arange(-1,1,0.0001)
y=np.sin(math.pi * x) #target function

#plt.plot(x,y , color='r')
a_s=[]
bias_arr=[]
''' we will use linear regression as it is built on minimizing the mean square error'''
for N in range (1000):
    
    data = np.random.uniform(-1., 1., size=(2,1)) # inputs X1, X2 
    output_TF =np.sin(math.pi * data )  # output of the target function 
    X_pseudo= inv((data.T).dot(data)).dot(data.T)
    a=X_pseudo.dot(output_TF)
    a_s.append(a)

g_bar=np.mean(a_s) # cooffecient a in g_bar=a*x 
print(g_bar)    

'''data = np.random.uniform(-1., 1., size=(2,1))
bias = (math.pow( (g_bar*data[0] - np.sin(math.pi * data[0] )),2 ) +  math.pow( (g_bar*data[1] - np.sin(math.pi * data[1] )),2 ) )/2
print("bias", bias)'''

''' plotting just for fun'''
#plt.scatter(data, output_TF)
line_x1 = np.linspace(-1, 1, 20000)
line_x2 = g_bar * line_x1 
#plt.plot(line_x1, line_x2) 

''' trying to calculate the bias ''' 

bias = np.mean(np.square( (g_bar*x) - y))
print("bias is ", bias)

''' desparate trial to calculate the varience as defined in lec 8 slide 9 '''
var_arr=[]
for N in range (10000):
    data = np.random.uniform(-1., 1., size=(2,1)) # inputs X1, X2 
    output_TF =np.sin(math.pi * data )  # output of the target function 
    X_pseudo= inv((data.T).dot(data)).dot(data.T)
    a=X_pseudo.dot(output_TF)
    var_D=np.mean(np.square( (a[0][0]*data)-(g_bar* data) ))
    var_arr.append(var_D)
    
var_final=np.mean(var_arr)
print("variance is",var_final)

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 22:32:28 2017

@author: Mrskhawy
"""


import numpy as np 
import matplotlib.pyplot as plt
import math 
import time
from numpy.linalg import inv

start_time = time.time()

#x=np.arange(-1,1,0.0001)
#y=np.sin(math.pi * x) #target function
#plt.plot(x,y , color='r')
''' we will use linear regression as it is built on minimizing the mean square error'''
''' using linear regression to solve for w in different cases as given in problem 7 from a to e'''
def avg_hypothesis_linear_reg(option):
    a_s=[]
    N_runs=1000
    for N in range (N_runs):
        data = np.random.uniform(-1., 1., size=(2,1)) # 2 training examples X1, X2 
        output_TF =np.sin(math.pi * data )  # output of the target function 
        if option == 1:
            data_mpltd=np.ones((2,1))
        elif option == 2:
            data_mpltd=data
        elif option == 3:
            data_mpltd= np.column_stack((np.ones(2), data))
        elif option == 4:
            data_mpltd=np.square(data)
        elif option == 5:
            data_mpltd= np.column_stack((np.ones(2), np.square(data)))
        X_pseudo= inv((data_mpltd.T).dot(data_mpltd)).dot(data_mpltd.T)
        a=X_pseudo.dot(output_TF)
        a_s.append(a)
    if option == 3 or option == 5:
        #reshaping the array to about the 3rd dimension when adding item to the list
        #now we have coulmn for b and column for a
        return np.reshape(a_s, (N_runs,2))  
    else : 
        return a_s

''' trying to calculate the bias using formula on lec8 slide 9''' 
def calc_bias(g_bar,option):
    x=np.arange(-1,1,0.0001)
    y=np.sin(math.pi * x) #target function
    if option == 1:
        y_g_bar=g_bar
    elif option == 2:
        y_g_bar=g_bar*x
    elif option == 3:
        y_g_bar=g_bar[1]*x + g_bar[0]  #h(x)=ax+b, a=g_bar[1], b = g_bar[0]
    elif option == 4:
        y_g_bar=g_bar*np.square(x)
    elif option == 5:
        y_g_bar=g_bar[1]*np.square(x) + g_bar[0] #h(x)=ax^2+b, a=g_bar[1], b = g_bar[0]
    
    bias = np.mean(np.square(y_g_bar -y))
    return bias

''' calculate the varience as defined in lec 8 slide 9 
    obtaining Expected value with respect to data (2 examples at a time) for multiple runs, then after finishing
    calculate expected value with respect to X (number of "2" examples generated)'''
def calc_variance(g_bar,option):
    var_arr=[]
    for N in range (1000):
        data = np.random.uniform(-1., 1., size=(2,1)) # inputs X1, X2 
        output_TF =np.sin(math.pi * data )  # output of the target function 
        if option == 1:
            data_mpltd=np.ones((2,1))
        elif option == 2:
            data_mpltd=data
        elif option == 3:
            data_mpltd= np.column_stack((np.ones(2), data))
        elif option == 4:
            data_mpltd=np.square(data)
        elif option == 5:
            data_mpltd= np.column_stack((np.ones(2), np.square(data)))

        X_pseudo= inv((data_mpltd.T).dot(data_mpltd)).dot(data_mpltd.T)
        w=X_pseudo.dot(output_TF)
        if option == 1:
            y_g_bar=g_bar
            y_hypothesis=w
        elif option == 2:
            y_g_bar=g_bar*data
            y_hypothesis=w*data
        elif option == 3:
            y_g_bar= g_bar*data_mpltd#g_bar[1]*data + g_bar[0]  #h(x)=ax+b, a=g_bar[1], b = g_bar[0]
            y_hypothesis=w*data_mpltd#w[1]*data + w[0]
        elif option == 4:
            y_g_bar=g_bar*np.square(data)
            y_hypothesis=w*np.square(data)
        elif option == 5:
            y_g_bar=g_bar*data_mpltd #h(x)=ax^2+b, a=g_bar[1], b = g_bar[0]
            y_hypothesis= w*data_mpltd#w[1]*np.square(data)+ w[0]        
        var_thisDataSet=np.mean(np.square( y_hypothesis - y_g_bar ))
        var_arr.append(var_thisDataSet)
    return var_arr
for option in range(1,6):
    
    g_bar=np.mean(avg_hypothesis_linear_reg(option),axis=0) # cooffecient a in g_bar=a*x 
    bias=calc_bias(g_bar,option)    
    var_final=np.mean(calc_variance(g_bar,option))
    E_out=bias+var_final
    #print("average hypothesis for hypotheis in %d, is %f =>" % (option, g_bar[0]))    
    print("bais is =>", bias)
    print("variance is =>",var_final)
    print("Eout for hypotheis in %d, is %f \n\n" %(option, E_out))

''' plotting just for fun
#plt.scatter(data, output_TF)
line_x1 = np.linspace(-1, 1, 20000)
line_x2 = g_bar * line_x1 
#plt.plot(line_x1, line_x2) '''


print("\n \n********Execution time is: %s seconds*******" % (time.time() - start_time))

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 04:00:08 2017

@author: Mrskhawy
"""

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
def avg_hypothesis_linear_reg_const():
    a_s=[]
    N_runs=10000
    for N in range (N_runs):
        data = np.random.uniform(-1., 1., size=(2,1)) # 2 training examples X1, X2 
        output_TF =np.sin(math.pi * data )  # output of the target function 
        data_mpltd=np.ones((2,1))
        X_pseudo= inv((data_mpltd.T).dot(data_mpltd)).dot(data_mpltd.T)
        a=X_pseudo.dot(output_TF)
        a_s.append(a)
        return a_s

''' trying to calculate the bias using formula on lec8 slide 9''' 
def calc_bias_const(g_bar):
    x=np.arange(-1,1,0.0001)
    y=np.sin(math.pi * x) #target function
    y_g_bar=g_bar
    bias = np.mean(np.square(y_g_bar -y))
    return bias

def calc_variance_const(g_bar):
    var_arr=[]
    for N in range (1000):
        data = np.random.uniform(-1., 1., size=(2,1)) # inputs X1, X2 
        output_TF =np.sin(math.pi * data )  # output of the target function 
        data_mpltd=np.ones((2,1))
        X_pseudo= inv((data_mpltd.T).dot(data_mpltd)).dot(data_mpltd.T)
        w=X_pseudo.dot(output_TF)
        y_g_bar=g_bar
        y_hypothesis=w
        var_thisDataSet=np.mean(np.square( y_hypothesis - y_g_bar ))
        var_arr.append(var_thisDataSet)
    return var_arr
    
'''g_bar=np.mean(avg_hypothesis_linear_reg_const(),axis=0) # cooffecient a in g_bar=a*x 
bias=calc_bias_const(g_bar)    
var_final=np.mean(calc_variance_const(g_bar))
E_out=bias+var_final
print("g_bar",g_bar)
print("bais is =>", bias)
print("variance is =>",var_final)
print("Eout ", E_out)'''

''' ********************************************************************** '''
def avg_hypothesis_linear_reg_aX_b():
    a_s=[]
    N_runs=1000
    for N in range (N_runs):
        data = np.random.uniform(-1., 1., size=(2,1)) # 2 training examples X1, X2 
        output_TF =np.sin(math.pi * data )  # output of the target function 
        data_mpltd= np.column_stack((np.ones(2), data))
        X_pseudo= inv((data_mpltd.T).dot(data_mpltd)).dot(data_mpltd.T)
        a=X_pseudo.dot(output_TF)
        #print(a)
        a_s.append(a)
    #print(np.reshape(a_s, (N_runs,2)))
    return np.reshape(a_s, (N_runs,2))  

def calc_bias_aX_b(g_bar):
    x=np.arange(-1,1,0.0001)
    y=np.sin(math.pi * x) #target function
    y_g_bar=g_bar[1]*x + g_bar[0]  #h(x)=ax+b, a=g_bar[1], b = g_bar[0]
    bias = np.mean(np.square(y_g_bar -y))
    return bias

def calc_variance_aX_b(g_bar):
    var_arr=[]
    for N in range (1000):
        data = np.random.uniform(-1., 1., size=(2,1)) # inputs X1, X2 
        output_TF =np.sin(math.pi * data )  # output of the target function 
        data_mpltd= np.column_stack((np.ones(2), data))
        '''print("data \n", data)
        print("data_mptld\n",data_mpltd)'''
        X_pseudo= inv((data_mpltd.T).dot(data_mpltd)).dot(data_mpltd.T)
        w=X_pseudo.dot(output_TF)
        y_g_bar=g_bar*data_mpltd#  #h(x)=ax+b, a=g_bar[1], b = g_bar[0]
        y_hypothesis= w*data_mpltd#w[1]*data + w[0]
        '''print("g_bar 1st == wrong\n",g_bar[1]*data + g_bar[0])
        print("g_bar_2nd== true \n",g_bar*data_mpltd)'''
        var_thisDataSet=np.mean(np.square( y_hypothesis - y_g_bar ))
        var_arr.append(var_thisDataSet)
   
    return var_arr

g_bar=np.mean(avg_hypothesis_linear_reg_aX_b(),axis=0) # cooffecient a in g_bar=a*x 
bias=calc_bias_aX_b(g_bar)    
var_final=np.mean(calc_variance_aX_b(g_bar))
E_out=bias+var_final
print("\n\n ******Hypothesis aX+b*******")
print("g_bar",g_bar)
print("bais is =>", bias)
print("variance is =>",var_final)
print("Eout ", E_out)


'''x=np.arange(-1,1,0.0001)
y=np.sin(math.pi * x) #target function
plt.plot(x,y , color='r')
line_x1 = np.linspace(-1, 1, 20000)
line_x2 = g_bar[1] * line_x1 + g_bar[0]
plt.plot(line_x1, line_x2)'''

print("\n \n********Execution time is: %s seconds*******" % (time.time() - start_time))

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 12:42:50 2017

@author: Mrskhawy
"""
import numpy as np
import random as random
from numpy.linalg import inv
import time

def generate_training_data(N):
    data = np.random.uniform(-1., 1., size=(N,2)) # N row (points) by 2 colums 
    data_th =  np.column_stack((np.ones(N), data)) # [1,x1,x2]    
    return data, data_th

def linear_regression(data_th, noisy_out,data_size):
        X_pseudo= inv((data_th.T).dot(data_th)).dot(data_th.T)
        W_g=X_pseudo.dot(noisy_out)
        ''' checking the data using the hypothesis '''
        output_g=classify_line(data_th,W_g)
        tmp=(output_g == output_TF)
        E_in=(np.size(tmp)-np.sum(tmp))/data_size
        return E_in

def lin_reg_nonlin_trans(Z, noisy_out,data_size):
        X_pseudo= inv((Z.T).dot(Z)).dot(Z.T)
        W_g=X_pseudo.dot(noisy_out)
        ''' checking the data using the hypothesis '''
        ''' output_g=classify_line(data_th,W_g)
        tmp=(output_g == output_TF)
        E_in=(np.size(tmp)-np.sum(tmp))/data_size'''
        return W_g


def classify(data):
    ''' f(x1,x2)=sign(x1^2 + x2^2 -0.6 '''
    output=np.sign(np.sum(data*data, axis=1)-0.6)
    return output

def classify_line(data_th,W):
    output=np.sign(data_th.dot(W))
    return output #output of target function       
    
def classify_nonlin(input_vec,w_vec,output_TF):
    output=np.sign(input_vec.dot(w_vec))
    tmp=(output == output_TF)
    E=(np.size(tmp)-np.sum(tmp))/1000
    return E #output of target function       
#number of data points
data_size=1000
E_in=[]
start_time = time.time()

for i in range (1000):
    
    data,data_th=generate_training_data(data_size)
    
    output_TF=classify(data)
    '''# copying the array not making a reference to it, not noisy_out = out_TF just makes 
    noisy_out reference to out_tf, so any change you do to noisy_out is done to out_tf'''
    noisy_out = np.copy(output_TF)
    ''' generates 0.1*N distinctive random numbers to use as indecis to toggel the output array'''
    rnd_indices=random.sample(range(0,data_size),100)
    noisy_out[rnd_indices]= noisy_out[rnd_indices] *(-1) #toggelling the value of the output at these rnd indices  
    E_in.append(linear_regression(data_th,noisy_out,data_size))
    
print("average Ein",np.mean(E_in))
''' using the last generated data points and noisy ouptut from the above loop'''
''' Non-Linear transformation for Q9 '''
tmp1=np.column_stack( (np.square(data[:,0]),np.square(data[:,1])) ) #[x1^2,x2^2]
tmp2=np.column_stack( (data_th, data[:,0]*data[:,1])      )       #[1,x1,x2,x1x2]
Z=np.column_stack((tmp2,tmp1)) 
W_non_lin = lin_reg_nonlin_trans(Z,noisy_out,data_size)

g1=np.array([-1,-0.05,0.08,0.13,1.5,1.5])
g2=np.array([-1,-0.05,0.08,0.13,1.5,15])
g3=np.array([-1,-0.05,0.08,0.13,15,1.5])
g4=np.array([-1,-1.5,0.08,0.13,0.05,0.05])
g5=np.array([-1,-0.05,0.08,1.5,0.15,0.15])
E_mine=[]
E_1=[]
E_2=[]
E_3=[]
E_4=[]
E_5=[]
''' this for loop to compare between hypothesis, un intentionally I solved Q10 here also by
generating fresh data, evaluating Tf for them then classifying using my hypothesis, i choose a for Q10 but it is wrong '''

for i in range (1000):    
    data,data_th=generate_training_data(data_size)
    output_TF=classify(data)
    noisy_out = np.copy(output_TF)
    rnd_indices=random.sample(range(0,data_size),100)
    noisy_out[rnd_indices]= noisy_out[rnd_indices] *(-1) 
    
    tmp1=np.column_stack( (np.square(data[:,0]),np.square(data[:,1])) ) #[x1^2,x2^2]
    tmp2=np.column_stack( (data_th, data[:,0]*data[:,1])      )       #[1,x1,x2,x1x2]
    Z=np.column_stack((tmp2,tmp1)) 
    E_mine.append(classify_nonlin(Z,W_non_lin, output_TF))
    E_1.append(classify_nonlin(Z,g1, output_TF))
    E_2.append(classify_nonlin(Z,g2, output_TF))
    E_3.append(classify_nonlin(Z,g3, output_TF))
    E_4.append(classify_nonlin(Z,g4, output_TF))
    E_5.append(classify_nonlin(Z,g5, output_TF))
''' #Z=(1,x1,x2,x1x2,x1^2,x2^2) '''


print ("E_mine",np.mean(E_mine))
print ("E_1",np.mean(E_1))
print ("E_2",np.mean(E_2))
print ("E_3",np.mean(E_3))
print ("E_4",np.mean(E_4))
print ("E_5",np.mean(E_5))


print("Execution time is: %s seconds" % (time.time() - start_time))

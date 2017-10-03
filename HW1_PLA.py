# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:45:13 2017

@author: Mrskhawy
"""
#Pick two random points and solve for the slope (m) and intercept (b) using the point slope formula
#Arrange the m and b in a weights vector equal to [b, m, -1]. 
#Choose new points with random values and a one at the beginning (dummy): [1, random(), random()]
#Take the dot product between the weights vector and new point vector

import random
import numpy as np
import time

def rnd():
    return random.uniform(-1, 1)

def classify(w_vec,x_vec):
    if np.dot(w_vec,x_vec) > 0: 
        return 1; 
    else:
        return 0; 
     
# Choose two random points A, B in the [-1,1] x [-1,1] box
start_time = time.time()
N_RUNS=1000
N=100
arr_N_iterations=[]
arr_mismatch=[]
for RUN_N in range(N_RUNS):

    P1 = [rnd(), rnd()]
    P2 = [rnd(), rnd()]
    m = (P2[1] - P1[1]) / (P2[0] - P1[0])  
    b = m * P2[0] - P2[1]
    w_vec=[b,m,-1]
    
#generating training examples 
    x1_points = np.array([rnd() for _ in range(N)])
    x2_points = np.array([rnd() for _ in range(N)])

    output_TF=[]  #output of target function
    #output of hypothesis g to diffrentiate between classified and misclassified
    #initially all points are misclassified
    output_g=[0 for x in range(N)]
#classifying them 
    for i in range(N):
        x1 = x1_points[i]
        x2 = x2_points[i]
        x_vec=[1,x1,x2];
        if classify(w_vec,x_vec) == 1:
            output_TF.append(1)
        else:
            output_TF.append(-1)
#we have our training points classified, it is time to implement the PLA 
    weight_g=[0,0,0] #initial , all the points missclassified
    N_iterations=0
    while(1): 
        #choosing a random point from the input set and check with the weights vector of g
        index=random.randint(0,N-1) 
        rnd_point=[1,x1_points[index], x2_points[index]]
        
        if classify(weight_g,rnd_point) == 1:
            output_g[index]=1
        else: 
            output_g[index]=-1
        
        if output_g[index] != output_TF[index]: 
        #misclassified point g disagrees with f, which means you have to update weight_g
        #weight_g= weight_g + output_TF[index]*(x1,x2) 
            weight_g= [x +y for x, y in zip(weight_g , [x*output_TF[index] for x in rnd_point]) ]
            #Updating the output_g for this particular point 
            output_g[index] = output_TF[index]
            N_iterations= N_iterations +1
        if output_g==output_TF: 
            arr_N_iterations.append(N_iterations)
            break

#calculating the disagreement 
    if False :        
        t=5000
        classification_mismatch=0
        
        x1_test_points = [rnd() for _ in range(t)]
        x2_test_points = [rnd() for _ in range(t)]
        for i in range(t):
            x1 = x1_test_points[i]
            x2 = x2_test_points[i]
            x_vec=[1,x1,x2];
            if  classify(w_vec,x_vec) != classify(weight_g, x_vec):
                classification_mismatch+=1
                arr_mismatch.append(classification_mismatch/t)
                

print("Average Number of Iterations to converge",sum(arr_N_iterations)/float(len(arr_N_iterations)))
#print("Probability of mismatch",sum(arr_mismatch)/float(len(arr_mismatch)))
print("Execution time is: %s seconds" % (time.time() - start_time))

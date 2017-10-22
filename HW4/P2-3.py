# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 03:26:57 2017

@author: Mrskhawy
"""
'''dvc=50 , mH(2N)= (2N)^dvc, delta=0.05 '''
import numpy as np 
import math 
import matplotlib.pyplot as plt
def nCr(n,r):
    f = math.factorial
    if r>n:
        return 0;
    else:
        return f(n) //( f(r) * f(n-r))

N=10000
#mH_N2=math.pow(N,100)
mH_2N=math.pow(2*N,50)
mH_N=math.pow(N,50)
delta=0.05

## VC bound 

eps_vc=math.sqrt((8/N)*np.log(4*mH_2N/delta))

#Rademacher Penalty Bound
eps_b=math.sqrt( (2/N)* np.log(2*N*mH_N) ) + math.sqrt( (2/N)* np.log(1/delta)  ) + (1/N)


# Parrondo and Van den Broek
''' with large N (10000 for ex) the ln term will be very large compared to that of eps, so we
can ignore the eps '''

eps_c=math.sqrt( (1/N)* np.log( (6/delta) * mH_2N ) )

#Devroye
''' with large N (10000 for ex) the ln term will be very large compared to that of eps, so we
can ignore the eps '''
eps_d=math.sqrt( (0.5/N)*( np.log(4/delta) + 100*np.log(N)) )

print("eps_vc",eps_vc)
print("eps_b",eps_b)
print("eps_c",eps_c)
print("eps_d",eps_d)
print("the right answer for question 2 is d")

eps_1=[math.sqrt((8/N)*np.log(4*math.pow(2*N,50)/0.05)) for N in range(1,10000)]
eps_2=[ math.sqrt( (2/N)* np.log(2*N*mH_N) ) + math.sqrt( (2/N)* np.log(1/delta)  ) + (1/N) for N in range(1,10000)]
eps_3=[math.sqrt( (1/N)* np.log( (6/delta) * mH_2N ) ) for N in range(1,10000)]

#plt.plot(range(1, 10000),eps_1 , color='r')
#plt.plot(range(1, 10000),eps_2,color='g' )
#plt.plot(range(1, 10000),eps_3, color='b' )
#plt.show()

''' Q3, solving iteratively for eps in option [c], [d] as N is small and you can't ignore eps this time'''
''' for small N the growth function is no more N^dvc, we have to use the original 
definition for growth function present in lec7 slide 4, or instead check lec7 slide 3

The VC dimension ofa hypothesis set H,denoted by dv(H),is the largest value of N for which
mH(N) = 2^N'''

from scipy.optimize import fsolve
N=5
dvc=10
delta=0.05

mH_N =np.sum([nCr(N,i) for i in range(dvc+1)])
mH_2N=np.sum([nCr(2*N,i) for i in range(dvc+1)])
mH_N2=33554432 #np.sum([nCr(N*N,i) for i in range(dvc+1)])

def equ_c(eps):
    tmp1=np.log( (6/delta) * mH_2N)
    return (math.pow(eps,2) - ( (1/N) *(2*eps + tmp1 )   )) 
def equ_d(eps): 
    return (math.pow(eps,2) - ( (0.05/N) * ( 4*eps*(1+eps) + np.log(4/delta) +np.log(mH_N2)) ) )
''' eps_c_iter=fsolve(equ_c,0.2 )
eps_d_iter=fsolve(equ_d, 0.3)
print("eps_c__iter, f",eps_c_iter,equ_c(eps_c_iter))
print("eps_d_iter , f",eps_d_iter,equ_d(eps_d_iter))
'''

import cmath
def solve_quad(a,b,c):
    delta = (b**2) - (4*a*c)
    solution1 = (-b-cmath.sqrt(delta))/(2*a)
    solution2 = (-b+cmath.sqrt(delta))/(2*a)
    return solution1, solution2

tmp1=np.log( (6/delta) * mH_2N)/N
epsC_0,epsC_1=solve_quad(1,-2/5,-1*tmp1)
print("epsC_0,epsC_1",epsC_0,epsC_1)
print(equ_c(1.7439535969958098))
eps_c_iter=1.7439535969958098

tmp2=np.log( (4/delta) * mH_N2)/(2*N)
epsD_0,epsD_1=solve_quad((1- (2/N)),-2/5,-1*tmp2)
print("epsD_0,epsD_1",epsD_0,epsD_1)
print(equ_d(2.264540762867992))
eps_d_iter=2.264540762867992
''' after obtaining good values for eps for the option c and d, now calculate the bound for small N'''

''' Note: value of eps_d_iter doesn't satisfy the equation
    I got the answer for Q3 wrong!!!!!!!!!!!!'''
eps_vc_small=math.sqrt((8/N)*np.log(4*mH_2N/delta))

#Rademacher Penalty Bound
eps_b_small=math.sqrt( (2/N)* np.log(2*N*mH_N) ) + math.sqrt( (2/N)* np.log(1/delta)  ) + (1/N)

eps_c_small=math.sqrt(( (1/N) *(2*eps_c_iter + np.log( (6/delta) * mH_2N )   )) )

eps_d_small= math.sqrt(( (0.05/N) * ( 4*eps_d_iter*(1+eps_d_iter) + np.log(4/delta) + np.log(mH_N2)) ) )

print("eps_vc_small",eps_vc_small)
print("eps_b_small",eps_b_small)
print("eps_c_small",eps_c_small)
print("eps_d_small",eps_d_small)

'''
eps_vc_small=[math.sqrt((8/N)*np.log(4*mH_2N/delta)) for N in range(1,10) ]

eps_b_small=[math.sqrt( (2/N)* np.log(2*N*mH_N) ) + math.sqrt( (2/N)* np.log(1/delta)  ) + (1/N) for N in range(1,10) ]

eps_c_small=[math.sqrt(( (1/N) *(2*eps_c_iter + np.log( (6/delta) * mH_2N )   )) ) for N in range(1,10) ]

eps_d_small=[ math.sqrt(( (0.05/N) * ( 4*eps_d_iter*(1+eps_d_iter) + np.log(4/delta) + 100*np.log(N)) ) ) for N in range(1,10) ]


plt.plot(range(1, 10),eps_vc_small , color='r')
plt.plot(range(1, 10),eps_b_small,color='g' )
plt.plot(range(1, 10),eps_c_small, color='b' )
plt.plot(range(1, 10),eps_d_small, color='m' )
plt.show()
'''
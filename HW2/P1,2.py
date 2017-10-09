
import numpy as np
import time

def flip_fair_coins(number_of_coins, N_flips):
    return np.random.randint(0,2,size=[number_of_coins,N_flips])

start_time = time.time()

N_exper=100000
N_coins=1000
N_flips=10
eps=np.arange(0.1,0.6,0.1) #eps can go up to 0.5
Mu=0.5
fraction_heads=np.array([])
nuMin_frac_heads=[]
nu1_frac_heads=[]
P_LHS_Hoeff_nu1=[]
P_LHS_Hoeff_nuRand=[]
P_LHS_Hoeff_nuMin=[]
'''fraction of heads is the relative frequency, which approximates the theoretical prob of event 
#happening when the number of experiments is large, ie 100000 like here '''
nuRand_frac_heads=[]
for k in range (N_exper) :    
    out=flip_fair_coins(N_coins, N_flips)
    #fraction_heads=[np.sum(out[i])/float(N_flips) for i in range(N_coins)]
    fraction_heads=np.mean(out,axis=1) # obtaining the mean for each row 
    nu1_frac_heads.append(fraction_heads[0]) #fraction_heads for the 1st coin flipped
    #generating random int as index to obtain nuRand
    nuRand_frac_heads.append(fraction_heads[np.random.randint(0,N_coins)])
    nuMin_frac_heads.append(min(fraction_heads))
#print(fraction_heads)
RHS_Hoeffding=2*np.exp(-2*eps*eps*N_flips)
'''calculating the LHS of Hoeffding inequality for each value of eps
   to get the P(A) where A is our event, you run the experiment for large number of times 
   100,000 here you count how many times the event A happens in the whole runs, then 
   P(A)=number of event happening/Number of runs 
   A in our case here is the |Nu-Mu|>eps. so we use the 100,000 values for each of nu1,nuRand
   and nuMin and count how many time the |Nu-Mu| is > eps, the devide by N_exper to get 
   P[|Nu-Mu|>eps], do that for each value of eps, then compare with the RHS ( 2e(-2eps*eps*N))
   
   As Mu=0.5 (probability of head) here, Nu (which should be approximating Mu) can take a min
   0 and max 1, so the error between them will be max 0.5, so eps can go from 0.1 up to 0.5
'''
for j in range(5):
    ''' |Nu-MU| > eps returns an array with values false and true, np.sum sums the number of
    True values (True=1, False=0) that's what it sums, if 3 Trues we get 3 and so on
    '''
    P_LHS_Hoeff_nu1.append(np.sum(abs( np.array(nu1_frac_heads)-Mu) > eps[j])/N_exper)
    P_LHS_Hoeff_nuRand.append(np.sum(abs( np.array(nuRand_frac_heads)-Mu) > eps[j])/N_exper)
    P_LHS_Hoeff_nuMin.append( np.sum(abs( np.array(nuMin_frac_heads)-Mu) > eps[j])/N_exper)

print(np.mean(nuMin_frac_heads))
print("Hoeffding inequality satisfiction for nu1", P_LHS_Hoeff_nu1 <= RHS_Hoeffding)
print("Hoeffding inequality satisfiction for nuRand", P_LHS_Hoeff_nuRand <= RHS_Hoeffding)
print("Hoeffding inequality satisfiction for nuMin", P_LHS_Hoeff_nuMin <= RHS_Hoeffding)

#plt.figure(figsize=(8, 4))
#plt.plot(range(1, N_exper+1), nu1_frac_heads)

print("Execution time is: %s seconds" % (time.time() - start_time))

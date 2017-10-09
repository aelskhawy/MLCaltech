
import numpy as np
import time
from numpy.linalg import inv


def generate_line(): 
    '''
    W dot X= 0 
    W0+W1X1+W2X2=0 
    
    choose two points (x1,x2) & (x1,x2) construct 2 equations and solve
    '''
    w0 = 1
    P1 , P2 = np.random.uniform(-1., 1., size=(2,2))
    # A*W = b 
    A = [P1 , P2]
    b= [-w0,-w0]
    w1, w2 = np.linalg.solve(A, b)
    return np.array([w0, w1, w2])

def generate_training_data(N):
    data = np.random.uniform(-1., 1., size=(N,2)) # N row (points) by 2 colums 
    data_set =  np.column_stack((np.ones(N), data)) # [1,x1,x2]    
    return data_set

def classify(points,W):
    output=np.sign(points.dot(W))
    return output #output of target function       
    
start_time = time.time()
N_data=100
N_fresh_data=1000
N_RUNS=1000
E_in=[]
E_out=[]
''' to keep both f's and g's  '''
TFs=[]
Gs=[]
def linear_regression(RUNS=1000, N=100):#default args values if no arguments are passed
    
    for i in range (RUNS):
        line=generate_line()
        TFs.append(line) #to keep the generated target functions 
        data_set= generate_training_data(N)
        output_TF=classify(data_set,line)
        #for matrix mult you have to use A dot AT not A*A as the later is element wise mult 
        X_pseudo= inv((data_set.T).dot(data_set)).dot(data_set.T)
        W_g=X_pseudo.dot(output_TF)
        Gs.append(W_g)
        ''' checking the data using the hypothesis '''
        output_g=classify(data_set,W_g)
        tmp=(output_g == output_TF)
        E_in.append((np.size(tmp)-np.sum(tmp))/N_data)
    
    return E_in, TFs, Gs

def PLA(TFs, Gs,RUNS =1000, N=10):
    arr_N_iterations=[]
    N_iterations=0
    for i in range(RUNS):
        ''' using the hypothesis from Linear Reg as intial weights to PLA'''
        N_iterations=0
        W_PLA=Gs[i]
        #print("W_PLA",W_PLA)
        data_set= generate_training_data(N)
        out_TF=classify(data_set,TFs[i])
        #print("out_TF",out_TF)
        while (1):
            out_G=classify(data_set,W_PLA)      
            #print(out_G)
            miss_class_indecis=np.where(out_G != out_TF)    
            #print("miss_class_indecis",miss_class_indecis)
            #print("size miss",np.size(miss_class_indecis))
            if np.size(miss_class_indecis) :
                ''' if miss_class_indecis has values, i.e missclassified points, then update W'''
                rnd_index=np.random.choice(miss_class_indecis[0])
                W_PLA= W_PLA + out_TF[rnd_index]*data_set[rnd_index]
                #print("WPLA after modification", W_PLA)
                N_iterations+=1
                #print("N_iter",N_iterations)
            else: 
                arr_N_iterations.append(N_iterations)
                break
            
    return arr_N_iterations

E_in, TFs, Gs= linear_regression()       

print("Average Ein after 1000 Run", np.mean(E_in))
print("Right answer for question 5 is", 0.01)


''' Q6: Generating fresh data points to check out of sample error (Eout)'''
for i in range (N_RUNS):
    fresh_data=generate_training_data(N_fresh_data)
    out_TF_fresh=classify(fresh_data,TFs[i])
    
    out_g_fresh=classify(fresh_data, Gs[i])
    tmp=(out_g_fresh == out_TF_fresh)
    E_out.append((np.size(tmp)-np.sum(tmp))/N_fresh_data)

print("Average Eout after 1000 Run", np.mean(E_out))
print("Right answer for Q6 is", 0.01)

''' Q7 '''
tmp=PLA(TFs, Gs,RUNS=1000,N=10)
print("Average Number of Iterations for PLA", np.mean(tmp))
print("Execution time is: %s seconds" % (time.time() - start_time))

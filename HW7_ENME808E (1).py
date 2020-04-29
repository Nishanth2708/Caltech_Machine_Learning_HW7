#!/usr/bin/env python
# coding: utf-8

# # The necessery explanation and code for the assignment https://work.caltech.edu/homework/hw7.pdf is provided below
# 
# { 
# 
#   1. **[d]**
#   2. **[e]**
#   3. **[d]**
#   4. **[d]**
#   5. **[b]**
#   6. **[d]**
#   7. **[c]**
#   8. **[c]**
#   9. **[d]**
#   10. **[b]** 
#  
#  }
# 

# In[351]:



import numpy as np
import math
import random
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import matrix, solvers
from sklearn.svm import SVC


# In[267]:


train_data = np.loadtxt('http://work.caltech.edu/data/in.dta')
test_data = np.loadtxt('http://work.caltech.edu/data/out.dta')


# In[268]:


x1, x2 = train_data[:25,0], train_data[:25,1]
X1, X2 = test_data[:,0], test_data[:,1]

val1, val2 = train_data[25:,0],  train_data[25:,1]


# In[269]:


Y_train = train_data[:25, 2]
Val_train = train_data[25:, 2]
Y_test = test_data[:,2]


# In[270]:


def transform7(x1, x2):
    
    shape = len(x1)
    k = 7
    data = np.zeros([shape, k+1])
    data[:,0] = 1
    data[:,1] = x1
    data[:,2] = x2
    data[:,3] = x1**2
    data[:,4] = x2**2
    data[:,5] = x1*x2
    data[:,6] = abs(x1-x2)
    data[:,7] = abs(x1+x2)
    
    return data
    


# In[271]:


def transform6(x1, x2):
    
    shape = len(x1)
    k = 6
    data = np.zeros([shape, k+1])
    data[:,0] = 1
    data[:,1] = x1
    data[:,2] = x2
    data[:,3] = x1**2
    data[:,4] = x2**2
    data[:,5] = x1*x2
    data[:,6] = abs(x1-x2)
#     data[:,7] = abs(x1+x2)
    
    return data
    


# In[272]:


def transform5(x1, x2):
    
    shape = len(x1)
    k = 5
    data = np.zeros([shape, k+1])
    data[:,0] = 1
    data[:,1] = x1
    data[:,2] = x2
    data[:,3] = x1**2
    data[:,4] = x2**2
    data[:,5] = x1*x2
#     data[:,6] = abs(x1-x2)
#     data[:,7] = abs(x1+x2)
    
    return data
    


# In[273]:


def transform4(x1, x2):
    
    shape = len(x1)
    k = 4
    data = np.zeros([shape, k+1])
    data[:,0] = 1
    data[:,1] = x1
    data[:,2] = x2
    data[:,3] = x1**2
    data[:,4] = x2**2
#     data[:,5] = x1*x2
#     data[:,6] = abs(x1-x2)
#     data[:,7] = abs(x1+x2)
    
    return data
    


# In[274]:


def transform3(x1, x2):
    
    shape = len(x1)
    k = 3
    data = np.zeros([shape, k+1])
    data[:,0] = 1
    data[:,1] = x1
    data[:,2] = x2
    data[:,3] = x1**2
#     data[:,4] = x2**2
#     data[:,5] = x1*x2
#     data[:,6] = abs(x1-x2)
#     data[:,7] = abs(x1+x2)
    
    return data
    


# In[275]:


test_3 = transform3(X1,X2)
test_4 = transform4(X1,X2)
test_5= transform5(X1,X2)
test_6= transform6(X1,X2)
test_7= transform7(X1,X2)


# # Question 1 & 2

# In[336]:


first_weight_term_3 = np.linalg.inv(np.dot(np.transpose(transform3(x1,x2)), transform3(x1,x2)))
second_weight_term_3 = np.dot(first_weight_term, np.transpose(transform3(x1,x2)))

weight_3 = np.dot(second_weight_term_3 , Y_train)
valdata_3= transform3(val1,val2)
predictval3 = np.sign(np.dot( valdata_3, weight_3))
predictval3_test = np.sign(np.dot(test_3, weight_3))

E_val3= np.mean(predictval3 != Val_train)
E_out3 = np.mean(predictval3_test != Y_test)
print('The Classification error for k=3 is ', E_val3)
print('The E_out for K= 3 is', E_out3)


# In[335]:


first_weight_term_4 = np.linalg.inv(np.dot(np.transpose(transform4(x1,x2)), transform4(x1,x2)))
second_weight_term_4 = np.dot(first_weight_term_4, np.transpose(transform4(x1,x2)))

weight_4 = np.dot(second_weight_term_4 , Y_train)
valdata_4= transform4(val1,val2)
predictval4 = np.sign(np.dot( valdata_4, weight_4))
predictval4_test = np.sign(np.dot(test_4, weight_4))

E_val4= np.mean(predictval4 != Val_train)
E_out4 = np.mean(predictval4_test != Y_test)


print('The Classification error for k=4 is ', E_val4)
print('The E_out for K= 4 is', E_out4)


# In[337]:


first_weight_term_5 = np.linalg.inv(np.dot(np.transpose(transform5(x1,x2)), transform5(x1,x2)))
second_weight_term_5 = np.dot(first_weight_term_5, np.transpose(transform5(x1,x2)))

weight_5 = np.dot(second_weight_term_5 , Y_train)
valdata_5= transform5(val1,val2)
predictval5 = np.sign(np.dot( valdata_5, weight_5))
E_val5= np.mean(predictval5 != Val_train)

predictval5_test = np.sign(np.dot(test_5, weight_5))
E_out5 = np.mean(predictval5_test != Y_test)

# print(E_val5)
# print(E_out5)

print('The Classification error for k=5 is ', E_val5)
print('The E_out for K= 5 is', E_out5)


# In[338]:


first_weight_term_6 = np.linalg.inv(np.dot(np.transpose(transform6(x1,x2)), transform6(x1,x2)))
second_weight_term_6 = np.dot(first_weight_term_6, np.transpose(transform6(x1,x2)))

weight_6 = np.dot(second_weight_term_6 , Y_train)
valdata_6= transform6(val1,val2)
predictval6 = np.sign(np.dot( valdata_6, weight_6))
E_val6= np.mean(predictval6 != Val_train)


predictval6_test = np.sign(np.dot(test_6, weight_6))
E_out6 = np.mean(predictval6_test != Y_test)

# print(E_val6)
# print(E_out6)

print('The Classification error for k=6 is ', E_val6)
print('The E_out for K= 6 is', E_out6)


# In[339]:


first_weight_term_7 = np.linalg.inv(np.dot(np.transpose(transform7(x1,x2)), transform7(x1,x2)))
second_weight_term_7 = np.dot(first_weight_term_7, np.transpose(transform7(x1,x2)))

weight_7 = np.dot(second_weight_term_7 , Y_train)
valdata_7= transform7(val1,val2)
predictval7 = np.sign(np.dot( valdata_7, weight_7))
E_val7= np.mean(predictval7 != Val_train)


predictval7_test = np.sign(np.dot(test_7, weight_7))
E_out7 = np.mean(predictval7_test != Y_test)

# print(E_val7)
# print(E_out7)

print('The Classification error for k=7 is ', E_val7)
print('The E_out for K= 7 is', E_out7)


# # Question 3 & 4

# In[353]:


first_weight_term_3_rev = np.linalg.inv(np.dot(np.transpose(transform3(val1,val2)), transform3(val1,val2)))
second_weight_term_3_rev = np.dot(first_weight_term_3_rev, np.transpose(transform3(val1,val2)))

weight_3_rev = np.dot(second_weight_term_3_rev , Val_train)
valdata_3_rev= transform3(x1,x2)
predictval3_rev = np.sign(np.dot( valdata_3_rev, weight_3_rev))
predictval3_test_rev = np.sign(np.dot(test_3, weight_3_rev))

E_val3_rev= np.mean(predictval3_rev != Y_train)
E_out3_rev = np.mean(predictval3_test_rev != Y_test)
# print(E_val3_rev)
# print(E_out3_rev)

print('The Classification error for k=3 is ', E_val3_rev)
print('The E_out for K= 3 is', E_out3_rev)


# In[340]:


first_weight_term_4_rev = np.linalg.inv(np.dot(np.transpose(transform4(val1,val2)), transform4(val1,val2)))
second_weight_term_4_rev = np.dot(first_weight_term_4_rev, np.transpose(transform4(val1,val2)))

weight_4_rev = np.dot(second_weight_term_4_rev , Val_train)
valdata_4_rev= transform4(x1,x2)
predictval4_rev = np.sign(np.dot( valdata_4_rev, weight_4_rev))
predictval4_test_rev = np.sign(np.dot(test_4, weight_4_rev))

E_val4_rev= np.mean(predictval4_rev != Y_train)
E_out4_rev = np.mean(predictval4_test_rev != Y_test)
# print(E_val4_rev)
# print(E_out4_rev)


print('The Classification error for k=4 is ', E_val4_rev)
print('The E_out for K= 4 is', E_out4_rev)


# In[354]:


first_weight_term_5_rev = np.linalg.inv(np.dot(np.transpose(transform5(val1,val2)), transform5(val1,val2)))
second_weight_term_5_rev = np.dot(first_weight_term_5_rev, np.transpose(transform5(val1,val2)))

weight_5_rev = np.dot(second_weight_term_5_rev , Val_train)
valdata_5_rev= transform5(x1,x2)
predictval5_rev = np.sign(np.dot( valdata_5_rev, weight_5_rev))
predictval5_test_rev = np.sign(np.dot(test_5, weight_5_rev))

E_val5_rev= np.mean(predictval5_rev != Y_train)
E_out5_rev = np.mean(predictval5_test_rev != Y_test)
# print(E_val5_rev)
# print(E_out5_rev)


print('The Classification error for k=5 is ', E_val5_rev)
print('The E_out for K= 5 is', E_out5_rev)


# In[341]:


first_weight_term_6_rev = np.linalg.inv(np.dot(np.transpose(transform6(val1,val2)), transform6(val1,val2)))
second_weight_term_6_rev = np.dot(first_weight_term_6_rev, np.transpose(transform6(val1,val2)))

weight_6_rev = np.dot(second_weight_term_6_rev , Val_train)
valdata_6_rev= transform6(x1,x2)
predictval6_rev = np.sign(np.dot( valdata_6_rev, weight_6_rev))
predictval6_test_rev = np.sign(np.dot(test_6, weight_6_rev))

E_val6_rev= np.mean(predictval6_rev != Y_train)
E_out6_rev = np.mean(predictval6_test_rev != Y_test)
# print(E_val6_rev)
# print(E_out6_rev)


print('The Classification error for k=6 is ', E_val6_rev)
print('The E_out for K= 6 is', E_out6_rev)


# In[342]:


first_weight_term_7_rev = np.linalg.inv(np.dot(np.transpose(transform7(val1,val2)), transform7(val1,val2)))
second_weight_term_7_rev = np.dot(first_weight_term_7_rev, np.transpose(transform7(val1,val2)))

weight_7_rev = np.dot(second_weight_term_7_rev , Val_train)
valdata_7_rev= transform7(x1,x2)
predictval7_rev = np.sign(np.dot( valdata_7_rev, weight_7_rev))
predictval7_test_rev = np.sign(np.dot(test_7, weight_7_rev))

E_val7_rev= np.mean(predictval7_rev != Y_train)
E_out7_rev = np.mean(predictval7_test_rev != Y_test)
# print(E_val7_rev)
# print(E_out7_rev)


print('The Classification error for k=7 is ', E_val7_rev)
print('The E_out for K= 7 is', E_out7_rev)


# #  Question 5

# For k =6, the out of sample error is least before and after reversing the train and validation sets the closest eucledian distance for 0.084, 0.192 is 0.1 and **0.2**

# # Question 6

# In[334]:


lst1 =[]
lst2=[]
lst3=[]
for e1 in range(300000):
    Y = ((random.random()))
    T = ((random.random()))
    lst1.append(Y)
    lst2.append(T)
    m = min(Y,T)
    lst3.append(m)
    
e_1= round((sum(lst1)/len(lst1)),3)
e_2 = round(sum(lst2)/len(lst2),3)
e3= round((sum(lst3)/len(lst3)),4)


print('After 300000 iterations the Expected value for \ne1 is {},\ne2 is {},\ne is {}'.format(e_1,e_2,e3))


# # Question 7

#  1. Considering the first model { h0(x) = b}, the line eqn passing from the points is observed to be 0 for which the square error from y= 0 to (rho, 1) is 1, Now, for the point (1,0) or (-1,0) and (rho,1) the line should be located in the middle of interval where y= 1/2. the validation error for two points would 1/4.
# 
#   **the total error is 1/3(1+ 2 * (1/4)) i.e, 1/2** -----------------> 1
# 
# 2. Considering the second model, {h(x) = ax+b}, the validation error for the points (-1, 0) and (1, 0) is 1. Now, for the same above points to the point the ( 2/ 1- rho) and 2/ ( 1+ rho).
# 
#   The total error is 1/3( 1 + ( 4/(1-rho )*2  +  ( 4/ (1+ rho)*2)
# 
#   error = 1/3 ( 1 + 4 / (1-rho)*2 + 4/(1+rho)*2 ) ---------------> 2
# 
#   equating equation 1 and 2
# 
#   1/2 = 1/3 ( 1 + 4 / (1-rho)*2 + 4/(1+rho)*2 )
#   1/2 =  4 / (1-rho)*2 + 4/(1+rho)*2
#   1/8 =  2(1+rho*2)/(1-rho*2)*4
# 
#   **rho^4 - 18* rho^2 - 15 = 0;
# 
#   After math, for rho >0, the value is **![image.png](attachment:image.png)**
# 

# # Question 8, 9 & 10

# In[355]:



# HW1 PLA###

def rnd(n): 
    return np.random.uniform(-1, 1, size = n)

RUNS = 1000
iterations_total = 0
ratio_mismatch_total = 0


################################## PLA ##########################################
for run in range(RUNS):

    # choose two random points A, B in [-1,1] x [-1,1]
    X = rnd(2)
    Y = rnd(2)

    m = (Y[1] - X[1]) / (Y[0] - X[0])
    b = Y[1] - m * X[0]  
    w_f = np.array([b, m, -1])

    N = 100
    X = np.transpose(np.array([np.ones(N), rnd(N), rnd(N)]))           # input
    y_f = np.sign(np.dot(X, w_f))                                      # output
    


    w_h = np.zeros(3)                       # initialize weight vector for hypothesis h
    t = 0                                   # count number of iterations in PLA
    
    while True:
        # Start PLA
        y_h = np.sign(np.dot(X, w_h))       # classification by hypothesis
        comp = (y_h != y_f)                 # compare classification with actual data from target function
        wrong = np.where(comp)[0]           # indices of points with wrong classification by hypothesis h

        if wrong.size == 0:
            break
        
        rnd_choice = np.random.choice(wrong)        # pick a random misclassified point

        # update weight vector (new hypothesis):
        w_h = w_h +  y_f[rnd_choice] * np.transpose(X[rnd_choice])
        t += 1

    iterations_total += t
    
    # Calculate error
    # Create data "outside" of training data

    N_outside = 1000
    test_x0 = np.random.uniform(-1,1,N_outside)
    test_x1 = np.random.uniform(-1,1,N_outside)

    X = np.array([np.ones(N_outside), test_x0, test_x1]).T

    y_target = np.sign(X.dot(w_f))
    y_hypothesis = np.sign(X.dot(w_h))
    
    ratio_mismatch = np.mean((y_target != y_hypothesis))
    ratio_mismatch_total += ratio_mismatch
    
print("Size of training data: N = ", N, "points")
    
iterations_avg = iterations_total / RUNS
print("\nAverage number of PLA iterations over", RUNS, "runs: t_avg = ", iterations_avg)

ratio_mismatch_avg = ratio_mismatch_total / RUNS
print("\nAverage ratio for the mismatch between f(x) and h(x) outside of the training data:")
print("P(f(x)!=h(x)) = ", ratio_mismatch_avg)


# In[327]:


def target_line():
    xa,ya,xb,yb = [random.uniform(-1, 1) for i in range(4)]
    weight = np.array([xb*ya-xa*yb, yb-ya, xa-xb])
    weight = weight.reshape((l.shape[0],1))
    f = lambda x:(yb-ya)/(xb-xa)*(x-xa)+ya
    return weight,f


# In[328]:


def xy_data(d,N,weight):        
    # Generate X and y values
    while True:
        X = np.column_stack((np.ones(N),
                         np.random.uniform(-1.,1., size=(N, d))))
        Y = np.array(np.where(np.dot(X,weight)>=0,1,-1))
        if Y.min() != Y.max(): # ensure two class target
            break
            
    return X,Y


# In[356]:



def pla(X, y, learn_rate):
            
    # Initialize weights at zero
    w = np.zeros((d+1,1))
        
    cnt_iter = 0
    while True:
        g = np.where(np.dot(X,w) >= 0, 1, -1)
        err = g - y
        misclass = np.nonzero(err)[0]
        if len(misclass) == 0:
            break
        else:
            point = random.choice(misclass) # choose a random point from the vector of misclassifications
            update = np.reshape(learn_rate*y[point]*X[point],(len(w),1))
            w = w + update 
            cnt_iter += 1
  
    
    return w


def test(Xt,yt,w):
    g = np.where(np.dot(Xt,w)>=0, 1, -1)
    E_out = np.count_nonzero(g - yt)/Nt
    return E_out


# ## Reference for SVM is taken from https://xavierbourretsicotte.github.io/SVM_implementation.html

# In[331]:



def svm_fitP(X,y):

    # Reshape y
    X = X[:,1:]
    y = y.reshape(-1,)
   
    # Get the necessary dimensions
    d = X.shape[1]
    L = d + 1
    M = X.shape[0]
    
    #=====PRIMAL FORMULATION=====
    # Compute required inputs of the problem
    p = np.zeros((L,1))
    Q = np.vstack([p.T,np.column_stack([p[1:],np.eye(d)])])
    A = -np.column_stack([y,y[:,None]*X])
    c = -np.ones((M))
    
    # Adjust the inputs to the algorithm used (see above notes)
    P = matrix(Q, tc = 'd')
    q = matrix(p, tc = 'd')
    G = matrix(A, tc = 'd')
    h = matrix(c, tc = 'd')

    # QP solution - primal problem
    solvers.options['show_progress'] = False
    w = solvers.qp(P, q, G, h)['x']
    
    return w

def svm_fitD(X,y):

    # Reshape y
    X = X[:,1:]
    y = y.reshape(-1,)
   
    # Get the necessary dimensions
    d = X.shape[1]
    L = d + 1
    M = X.shape[0]
        
    #=====DUAL FORMULATION=====
    # Compute required inputs of the problem
    y = y.reshape(-1,1)
    QD = (np.dot(y,y.T)) * (np.dot(X,X.T))
    P = matrix(QD, tc = 'd')
    AD = -np.vstack([y.T,-y.T,np.eye(M)])
    A = matrix (AD, tc = 'd')
    q = matrix(-np.ones((M,1)), tc = 'd')
    h = matrix(np.zeros((M+2)), tc='d')
    
    # QP solution - dual problem
    solvers.options['abstol'] = 1e-6
    solvers.options['reltol'] = 1e-5
    solvers.options['feastol'] = 1e-6
    sol = solvers.qp(P,q,A,h)
    alpha = np.around(np.array(sol['x']), decimals = 2)
    
    return alpha
        


# In[329]:



N = 10
Nt = 1000 
d = 2 
no_trials = 1000 
learn_rate = 0.01 


# In[362]:


Epla = []
Esvm = []
count = 0
for i in range(1000):
    
    weight,f = target_line()
    X,y = xy_data(2,10,l)
    Xt,yt = xy_data(d,Nt,l)

    w_pla = pla(X, y, learn_rate)
    E_pla = test(Xt,yt,w_pla)
    
    w_svm = svm_fitP(X,y)
    E_svm = test(Xt,yt, w_svm)
    
    Epla.append(E_pla)
    Esvm.append(E_svm)
    
    if E_svm < E_pla:
        count += 1
        
print('Percentage of times SVM is better equals to ', (count/float(no_trials))*100,'%')


# In[360]:


N = 100 # number of train examples
no_trials = 1000
d = 2

Epla = []
Esvm = []
count = 0
for i in range(no_trials):
    
    l,f = target_line()
    X,y = xy_data(d,N,l)
    Xt,yt = xy_data(d,Nt,l)

    w_pla = pla(X, y, learn_rate)
    E_pla = test(Xt,yt,w_pla)
    
    w_svm = svm_fitP(X,y)
    E_svm = test(Xt,yt, w_svm)
    
    Epla.append(E_pla)
    Esvm.append(E_svm)
    
    if E_svm < E_pla:
        count += 1
        
print('Percentage of times SVM is better equals to ', (count/float(no_trials))*100,'%')


# In[347]:


N = 100
no_trials = 500
d = 2

Epla = []
Esvm = []
sv_cnt = 0 # count support vetors per trial run
count = 0
for i in range(no_trials):
    
    l,f = target_line()
    X,y = xy_data(d,N,l)
    Xt,yt = xy_data(d,Nt,l)

    w_pla = pla(X, y, learn_rate)
    E_pla = test(Xt,yt,w_pla)
    
    alpha = svm_fitD(X,y)
    E_svm = test(Xt,yt, w_svm)
    sv_cnt += len(alpha[alpha>0])
    
    Epla.append(E_pla)
    Esvm.append(E_svm)
    
    if E_svm < E_pla:
        count += 1
        
print('Average number of support vectors per trial run is ', sv_cnt/float(no_trials))


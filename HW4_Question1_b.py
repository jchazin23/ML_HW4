# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 17:17:07 2016

@author: jchazin
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

ratings_fake = np.loadtxt('ratings_fake.csv',delimiter=',')

k = 5 #parameter for feature vector
T = 20 # number of iterations
mu = np.mean(ratings_fake[:,2])#average of A
m = 200
n = 200
b = np.zeros([m,1])
c = np.zeros([1,n])

#create a_i_j as a user x movie ratings matrix
a_i_j = np.reshape(ratings_fake[:,2],(200,200))

log_likelihood_theta = []

mean = np.zeros(k)
sig = (1.0/k) * np.identity(k)

u = np.random.multivariate_normal(mean,sig,(m))## initialize u_hat_i for each user: multiplied by 5 to mimic 1-5 rating scale
v = np.random.multivariate_normal(mean,sig,(n))## initialize v_hat_j for each movie: multiplied by 5 to mimic 1-5 rating scale

## in this iteration, I removed the mu from inside each summation, moving it outside to avoid
## duplication (since it is not subscripted by i or j)
for t in range(T):
    for i in range(m):
        left_u = 0
        right_u = 0 
        b_dummy = 0
        for j in range(n):
            left_u += -(np.outer(v[j,:],v[j,:].T))
            right_u += (-(b[i] + c[:,j] + mu - a_i_j[i,j]))*v[j,:]
            b_dummy += -(np.dot(u[i,:],v[j,:]) + c[:,j] + mu - a_i_j[i,j])
        u[i,:] = np.dot(inv(left_u),right_u)
        b[i] = b_dummy/n
            
    for j in range(n):
        left_v = 0
        right_v = 0 
        c_dummy = 0
        for i in range(m):
            left_v += -(np.outer(u[i,:],u[i,:].T))
            right_v += (-(b[i] + c[:,j] + mu - a_i_j[i,j]))*u[i,:]
            c_dummy += -(np.dot(u[i,:],v[j,:]) + b[i] + mu - a_i_j[i,j])
         
        v[j,:] = np.dot(inv(left_v),right_v)
        c[:,j] = c_dummy/m
    
    result = 0
    for i in range(m):
        for j in range(n):
            result += (np.dot(u[i,:],v[j,:]) + b[i] + c[:,j] + mu - a_i_j[i,j])
    result = (-0.5)*(result)**2
           
    log_likelihood_theta.append(result)

plt.plot(log_likelihood_theta)
plt.ylabel("Log Likelihood")
plt.xlabel("Iteration")
plt.title("HW 4, Question 1, Part B")
plt.show()


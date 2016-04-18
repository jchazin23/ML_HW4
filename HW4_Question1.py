# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 17:17:07 2016

@author: jchazin
"""

import numpy as np
import matplotlib.pyplot as plt

ratings_fake = np.loadtxt('ratings_fake.csv',delimiter=',')
ratings_test = np.loadtxt('ratings_test.csv',delimiter=',')
ratings_train = np.loadtxt('ratings_train.csv',delimiter = ',')


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

u = 5 * np.random.rand(m, k)## initialize u_hat_i for each user: multiplied by 5 to mimic 1-5 rating scale
v = 5 * np.random.rand(k, n)## initialize v_hat_j for each movie: multiplied by 5 to mimic 1-5 rating scale

## in this iteration, I removed the mu from inside each summation, moving it outside to avoid
## duplication (since it is not subscripted by i or j)
for t in range(T):
    for i in range(m):
        for j in range(n):
            u[i] = -(np.dot(v[:,j],v[:,j].T)**-1) * (-(b[i] - a_i_j[i,j]))*v[:,j]
            b[i] = -(np.dot(u[i],v[:,j]) + c[:,j] - a_i_j[i,j])/n
        u[i] -= mu
        b[i] -= mu            
            
    for j in range(n):
        for i in range(m):
            v[:,j] = -(np.dot(u[i],u[i].T)**-1) * (-(b[i] - a_i_j[i,j]))*u[i]
            c[:,j] = -(np.dot(u[i],v[:,j]) + b[i] - a_i_j[i,j])/m
        v[:,j] -= mu
        c[:,j] -= mu
    
    result = 0
    for i in range(m):
        for j in range(n):
            result += np.dot(u[i],v[:,j]) + b[i] + c[:,j] - a_i_j[i,j] 
    result = (-0.5)*(result + mu)**2
           
    log_likelihood_theta.append(result)

plt.plot(log_likelihood_theta)
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 17:17:07 2016

@author: jchazin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


ratings_train = pd.read_csv('ratings_train.csv',delimiter = ',',names=['User','Movie','Rating'])


## idea is to re-create a_i_j as a sparse matrix
A = np.zeros(shape=(942,1682))

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if ratings_train[(ratings_train.User == i+1)&(ratings_train.Movie ==j+1)].empty == False:
            A[i,j] = ratings_train[(ratings_train.User == i+1)&(ratings_train.Movie ==j+1)].iloc[:,2]
            


#train_users = ratings_train[:,0]
#train_movies = ratings_train[:,1]
#train_ratings = ratings_train[:,2]
#
#k = 10 #parameter for feature vector
#T = 5 # number of iterations
#mu = np.mean(ratings_train[:,2])#average of A
#m = 942
#n = 1682
#b = np.zeros([m,1])
#c = np.zeros([1,n])
#
#
#log_likelihood_theta = []
#
#u = 5 * np.random.rand(m, k)## initialize u_hat_i for each user: multiplied by 5 to mimic 1-5 rating scale
#v = 5 * np.random.rand(k, n)## initialize v_hat_j for each movie: multiplied by 5 to mimic 1-5 rating scale
#
### in this iteration, I removed the mu from inside each summation, moving it outside to avoid
### duplication (since it is not subscripted by i or j)
#for t in range(T):
#    for i in range(m):
#        for j in range(n):
#            #This checks whether this user/movie rating combo exists
#            if np.where(np.logical_and(train_users==m,train_movies==n))[0].shape[0] > 0:
#                a_i_j = ratings_train[np.where(np.logical_and(ratings_train[:,0]==m,ratings_train[:,1]==n)),2][0][0]
#                u[i] = -(np.dot(v[:,j],v[:,j].T)**-1) * (-(b[i] - a_i_j))*v[:,j]
#                b[i] = -(np.dot(u[i],v[:,j]) + c[:,j] - a_i_j)/n
#        u[i] -= mu
#        b[i] -= mu            
#            
#    for j in range(n):
#        for i in range(m):
#            #This checks whether this user/movie rating combo exists
#            if np.where(np.logical_and(train_users==m,train_movies==n))[0].shape[0] > 0:
#                a_i_j = ratings_train[np.where(np.logical_and(ratings_train[:,0]==m,ratings_train[:,1]==n)),2][0][0] 
#                v[:,j] = -(np.dot(u[i],u[i].T)**-1) * (-(b[i] - a_i_j))*u[i]
#                c[:,j] = -(np.dot(u[i],v[:,j]) + b[i] - a_i_j)/m
#        v[:,j] -= mu
#        c[:,j] -= mu
#    
#    result = 0
#    for i in range(m):
#        for j in range(n):
#            if np.where(np.logical_and(train_users==m,train_movies==n))[0].shape[0] > 0:
#                a_i_j = ratings_train[np.where(np.logical_and(ratings_train[:,0]==m,ratings_train[:,1]==n)),2][0][0] 
#                result += np.dot(u[i],v[:,j]) + b[i] + c[:,j] - a_i_j 
#    result = (-0.5)*(result + mu)**2
#           
#    log_likelihood_theta.append(result)
#
#plt.plot(log_likelihood_theta)
#plt.ylabel("Log Likelihood")
#plt.xlabel("Iteration")
#plt.title("HW 4, Question 1, Part C")
#plt.show()


# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 13:26:47 2019

@author: PRAJVAL
"""

import numpy as np
from numpy import matlib
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

data1 = np.genfromtxt("D:\IIT BOMBAY\Academics\Sem 4\GNR 652 Machine Learning for Remote Sensing-I\Coding Assignment 2\ones_only.csv", delimiter =',', skip_header=1)
data0 = np.genfromtxt("D:\IIT BOMBAY\Academics\Sem 4\GNR 652 Machine Learning for Remote Sensing-I\Coding Assignment 2\zeros_only.csv", delimiter =',', skip_header=1)

for i in range(0,5):
    
    random_indices0 = np.random.permutation(284333)
    random_indices1 = np.random.permutation(472)
    
    x_train0 = data0[random_indices0[:80]]
    x_test0 = data0[random_indices0[80:100]]
    x_train1 = data1[random_indices1[:80]]
    x_test1 = data1[random_indices1[80:100]]
    y_train1 = matlib.ones((80,1))
    y_train0 = y_train1*-1
    x_train0 = x_train0*-1
    x_test0 = x_test0*-1
    
    y_train = np.concatenate((y_train0,y_train1),axis=0)
    x_train = np.concatenate((x_train0,x_train1),axis=0)
    
    z=y_train.reshape(1,-1)
    
    H = np.matmul(x_train,x_train.transpose())
    m=160
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(m))
    A = cvxopt_matrix(z)
    b = cvxopt_matrix(np.zeros(1))
    
    sol = cvxopt_solvers.qp(P, q, G, h, A, b);
    alphas = np.array(sol['x'])
    
    w = np.matmul(alphas.T,x_train)
    b= np.min(w.dot(x_train1.T))+np.max(w.dot(x_train0.T))
    b=-0.5*b
    
    test1 = w.dot(x_test1.T) + b*np.ones((1,20))
    test2 = w.dot(x_test0.T) + b*np.ones((1,20))
    sum0=0; sum1=0;
    for i in range(20):
        if(test2[0,i]<0): sum0=sum0+1
        if(test1[0,i]>0): sum1=sum1+1
    
    sum = (sum0 + sum1)*2.5
    
    print("Accuracy: ", sum)
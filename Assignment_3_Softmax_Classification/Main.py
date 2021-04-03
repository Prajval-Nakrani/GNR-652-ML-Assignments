# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:00:02 2019

@author: PRAJVAL
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.io import loadmat
import numpy as np

X_data=loadmat("D:\IIT BOMBAY\Academics\Sem 4\GNR 652 Machine Learning for Remote Sensing-I\Coding Assignment 3\Indian_pines_corrected.mat")
X_data=X_data['indian_pines_corrected']
x_data=X_data.copy()

for j in range(x_data.shape[1]):
    maxv=np.max(x_data[:,j])
    minv=np.min(x_data[:,j])
    x_data[:,j]=(x_data[:,j]-maxv)/(maxv-minv)

Y_data=loadmat("D:\IIT BOMBAY\Academics\Sem 4\GNR 652 Machine Learning for Remote Sensing-I\Coding Assignment 3\Indian_pines_gt.mat")
Y_data = Y_data['indian_pines_gt']
y_data=Y_data.copy()

x_data = np.resize(x_data,(21025, 200))
y_data = np.resize(y_data,(21025,1))

dum = np.zeros((200,1))
dum1 = np.zeros((1,1))

x_data = x_data.T


print('Please wait while calculation we calculate Accuracy:')
for i in range (21025):
    if y_data[i-1,0] != 0 :
        l = x_data[:, i-1]
        k = np.resize(l, (200, 1))
        l1 = y_data[i-1, :]
        k1 = np.resize(l1, (1, 1))
        dum  = np.append(dum, k, axis=1)
        dum1 = np.append(dum1, k1, axis=1)

x_cleaned = dum[:,1:]
y_cleaned = dum1[:,1:]
x_cleaned = np.matrix(x_cleaned)
y_cleaned = np.matrix(y_cleaned)


x = np.ones((1,5249), int)
x_train = x_cleaned[:,5000:]
x_train = np.concatenate((x,x_train),axis=0)

y_train = y_cleaned[:,5000:]
y_train = y_train.astype(int)


x_test = x_cleaned[:,:5000]
x = np.ones((1,5000),int)
x_test = np.concatenate((x,x_test),axis=0)
y_test = y_cleaned[:,:5000]
y_test = y_test.T
y_test = y_test.astype(int)

C = np.zeros((5249,16))
C = np.matrix(C)
for s in range(5249):
        b =  y_train[0,s]
        C[s,b-1] = 1


w=np.zeros((201,16))
alpha = 0.0000001

for i in range(100):
    prob_mat = np.matmul(x_train.T,w)
    prob_mat = np.exp(prob_mat - np.max(prob_mat, axis=1))
    prob_mat = prob_mat-np.max(prob_mat)
    prob_mat = np.exp(prob_mat)
    sum_matrix = prob_mat.sum(axis=1)
    for r in range(5249):
        prob_mat[r,:] = prob_mat[r,:]/sum_matrix[r,0]
    m = C-prob_mat
    gradient = np.matmul(x_train,m)
    w = w - alpha*gradient
    

prob_mat1 = np.matmul(x_test.T,w)
prb_mat1 = np.exp(prob_mat1 - np.max(prob_mat1, axis=1))
prob_mat1 = np.exp(prob_mat1)
prob_mat1 = np.matrix(prob_mat1)
sum_matrix1 = prob_mat1.sum(axis=1)
for r in range(5000):
    prob_mat1[r,:] = prob_mat1[r,:]/sum_matrix1[r,0]
    
y_predicted = np.argmax(prob_mat1, axis=1)

match = 0
for i in range(5000):
    if(y_predicted[i,0] ==y_test[i,0]):
        match = match+1;

print("Accuracy = ", match/50,"%")
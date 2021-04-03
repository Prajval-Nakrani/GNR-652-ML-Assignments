# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 21:57:54 2019

@author: PRAJVAL
"""

import numpy.matlib
import numpy as np

data = np.genfromtxt("Data.csv", delimiter =',', skip_header=1)

x = np.ones((135,1), int)

data = np.concatenate((x,data),axis=1)

x=data[:,0:18]
y = data[:,18]

x = np.asmatrix(x)
y = np.asmatrix(y)
y = y.transpose()

x_train = x[0:108]
x_test = x[108:]
y_train = y[0:108]
y_test = y[108:]

def cost(X,Y,theta,lamda):
    predictions = X.dot(theta)
    error = predictions - Y
    error_sqr = error.transpose().dot(error)    
    return error_sqr

alpha = 0.00001
lamda = 0.01
iter_num = 15000

theta = np.matlib.zeros((18,1))

for i in range(iter_num):
    
    correction = x_train.transpose().dot((x_train.dot(theta) - y_train)) + lamda*theta
    theta = theta - alpha*correction
    
print ("First Run:\n")

print ("Parameters: \n", theta)

print ("\nTrain Variance: ", float(cost(x_train,y_train,theta,lamda)/108))

print ("Test Variance: ", float(cost(x_test,y_test,theta,lamda)/27))


"""SECOND RUN"""

random_indices = np.random.permutation(135)

x_train = x[random_indices[:108]]
x_test = x[random_indices[108:]]
y_train = y[random_indices[:108]]
y_test = y[random_indices[108:]]

theta = np.matlib.zeros((18,1))

for i in range(iter_num):
    
    correction = x_train.transpose().dot((x_train.dot(theta) - y_train)) + lamda*theta
    theta = theta - alpha*correction
    
print ("\n\nSecond Run:\n")

print ("Parameters: \n", theta)

print ("\nTrain Variance: ", float(cost(x_train,y_train,theta,lamda)/108))

print ("Test Variance: ", float(cost(x_test,y_test,theta,lamda)/27))


"""THIRD RUN"""

random_indices = np.random.permutation(135)

x_train = x[random_indices[:108]]
x_test = x[random_indices[108:]]
y_train = y[random_indices[:108]]
y_test = y[random_indices[108:]]

theta = np.matlib.zeros((18,1))

for i in range(iter_num):
    
    correction = x_train.transpose().dot((x_train.dot(theta) - y_train)) + lamda*theta
    theta = theta - alpha*correction
    
print ("\n\nThird Run:\n")

print ("Parameters: \n", theta)

print ("\nTrain Variance: ", float(cost(x_train,y_train,theta,lamda)/108))

print ("Test Variance: ", float(cost(x_test,y_test,theta,lamda)/27))

"""FOURTH RUN"""

random_indices = np.random.permutation(135)

x_train = x[random_indices[:108]]
x_test = x[random_indices[108:]]
y_train = y[random_indices[:108]]
y_test = y[random_indices[108:]]

theta = np.matlib.zeros((18,1))

for i in range(iter_num):
    
    correction = x_train.transpose().dot((x_train.dot(theta) - y_train)) + lamda*theta
    theta = theta - alpha*correction
    
print ("\n\nFourth Run:\n")

print ("Parameters: \n", theta)

print ("\nTrain Variance: ", float(cost(x_train,y_train,theta,lamda)/108))

print ("Test Variance: ", float(cost(x_test,y_test,theta,lamda)/27))

"""FIFTH RUN"""

random_indices = np.random.permutation(135)

x_train = x[random_indices[:108]]
x_test = x[random_indices[108:]]
y_train = y[random_indices[:108]]
y_test = y[random_indices[108:]]

theta = np.matlib.zeros((18,1))

for i in range(iter_num):
    
    correction = x_train.transpose().dot((x_train.dot(theta) - y_train)) + lamda*theta
    theta = theta - alpha*correction
    
print ("\n\nFifth Run:\n")

print ("Parameters: \n", theta)

print ("\nTrain Variance: ", float(cost(x_train,y_train,theta,lamda)/108))

print ("Test Variance: ", float(cost(x_test,y_test,theta,lamda)/27))
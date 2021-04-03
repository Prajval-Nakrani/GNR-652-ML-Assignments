# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 20:13:34 2019

@author: PRAJVAL
"""

import numpy as np
import numpy.matlib

data1 = np.genfromtxt("Data.csv", delimiter =',', skip_header=1)


"""norm_fact = [27,4,5,1,3,2,1,1,1,4,1,7,1,8,1,4,1,23.4]
norm_fact = np.array(norm_fact)
data2 = data2/norm_fact"""
"""
X_norm = data2
mu = np.zeros((1,18))
sigma = np.zeros((1,18))

for i in range(18):
    mu[0,i] = np.mean(data2[:,i])
    sigma[0,i] = np.std(data2[:,i])
    
    X_norm[:,i] = (data2[:,i]-mu[0,i])/sigma[0,i]

data2 = X_norm
"""
x = np.ones((135,1), int)

data1 = np.concatenate((x,data1),axis=1)

x=data1[:,0:18]
y = data1[:,18]

x = np.asmatrix(x)
y = np.asmatrix(y)
y = y.transpose()

x_train = data1[0:108,0:18]
x_test = data1[108:,0:18]
y_train = data1[0:108,18]
y_test = data1[108:,18]

X_train = np.asmatrix(x_train)
X_test = np.asmatrix(x_test)
Y = np.asmatrix(y_train)
Y_train = Y.transpose()
Y = np.asmatrix(y_test)
Y_test = Y.transpose()

lamda = 0.01

L = np.matlib.identity(18,int)
M = np.matlib.eye(18,1,0,int)
N = np.zeros((18,17))
L = L - np.concatenate((M,N),axis=1)

theta = np.linalg.inv(X_train.transpose().dot(X_train) + lamda*L).dot(X_train.transpose()).dot(Y_train)

theta2 = np.linalg.inv(x.transpose().dot(x) + lamda*L).dot(x.transpose()).dot(y)

def cost(X,Y,theta,lamda):
    predictions = X.dot(theta)
    error = predictions - Y
    error_sqr = error.transpose().dot(error)
    J = 0.5*error_sqr + 0.5*lamda*(theta.transpose().dot(theta))
    return J

print ("First Shuffle: \n")

print ("Parameters:\n", theta)

print ("\nTrain Cost: " , float(cost(X_train,Y_train,theta,lamda)/108))

print ("Test Cost: " , float(cost(X_test,Y_test,theta,lamda)/27))


"""SECOND RUN"""


random_indices = np.random.permutation(135)

x_train = x[random_indices[:108]]
x_test = x[random_indices[108:]]
y_train = y[random_indices[:108]]
y_test = y[random_indices[108:]]

X_train = np.asmatrix(x_train)
X_test = np.asmatrix(x_test)
Y = np.asmatrix(y_train)
Y = np.asmatrix(y_test)

theta = np.linalg.inv(X_train.transpose().dot(X_train) + lamda*L).dot(X_train.transpose()).dot(Y_train)

print ("\n\nSecond Shuffle: \n")

print ("Parameters:\n", theta)

print ("\nTrain Cost: " , float(cost(X_train,Y_train,theta,lamda)/108))

print ("Test Cost: " , float(cost(X_test,Y_test,theta,lamda)/27))

"""THIRD RUN"""


random_indices = np.random.permutation(135)

x_train = x[random_indices[:108]]
x_test = x[random_indices[108:]]
y_train = y[random_indices[:108]]
y_test = y[random_indices[108:]]

X_train = np.asmatrix(x_train)
X_test = np.asmatrix(x_test)
Y = np.asmatrix(y_train)
Y = np.asmatrix(y_test)

theta = np.linalg.inv(X_train.transpose().dot(X_train) + lamda*L).dot(X_train.transpose()).dot(Y_train)

print ("\n\nThird Shuffle: \n")

print ("Parameters:\n", theta)

print ("\nTrain Cost: " , float(cost(X_train,Y_train,theta,lamda)/108))

print ("Test Cost: " , float(cost(X_test,Y_test,theta,lamda)/27))

"""FOURTH RUN"""


random_indices = np.random.permutation(135)

x_train = x[random_indices[:108]]
x_test = x[random_indices[108:]]
y_train = y[random_indices[:108]]
y_test = y[random_indices[108:]]

X_train = np.asmatrix(x_train)
X_test = np.asmatrix(x_test)
Y = np.asmatrix(y_train)
Y = np.asmatrix(y_test)

theta = np.linalg.inv(X_train.transpose().dot(X_train) + lamda*L).dot(X_train.transpose()).dot(Y_train)

print ("\n\nFourth Shuffle: \n")

print ("Parameters:\n", theta)

print ("\nTrain Cost: " , float(cost(X_train,Y_train,theta,lamda)/108))

print ("Test Cost: " , float(cost(X_test,Y_test,theta,lamda)/27))

"""FIFTH RUN"""


random_indices = np.random.permutation(135)

x_train = x[random_indices[:108]]
x_test = x[random_indices[108:]]
y_train = y[random_indices[:108]]
y_test = y[random_indices[108:]]

X_train = np.asmatrix(x_train)
X_test = np.asmatrix(x_test)
Y = np.asmatrix(y_train)
Y = np.asmatrix(y_test)

theta = np.linalg.inv(X_train.transpose().dot(X_train) + lamda*L).dot(X_train.transpose()).dot(Y_train)

print ("\n\nFifth Shuffle: \n")

print ("Parameters:\n", theta)

print ("\nTrain Cost: " , float(cost(X_train,Y_train,theta,lamda)/108))

print ("Test Cost: " , float(cost(X_test,Y_test,theta,lamda)/27))
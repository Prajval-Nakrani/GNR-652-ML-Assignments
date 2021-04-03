# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 11:31:44 2019

@author: PRAJVAL
"""

import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt


data2 = np.genfromtxt("Data.csv", delimiter =',', skip_header=1)

x = np.ones((135,1), int)

data2 = np.concatenate((x,data2),axis=1)

x=data2[:,0:18]
y = data2[:,18]

x = np.asmatrix(x)
y = np.asmatrix(y)
y = y.transpose()

X_train = data2[0:108,0:18]
X_test = data2[108:,0:18]
Y_train = data2[0:108,18]
Y_test = data2[108:,18]


def cost(X,Y,theta,lamda):
    predictions = X.dot(theta)
    error = predictions - Y
    error_sqr = error.transpose().dot(error)
    J = 0.5*error_sqr + 0.5*lamda*(theta.transpose().dot(theta))
    
    return J

alpha = 0.00001
lamda = 0.01
iter_num = 15000

theta = np.matlib.zeros((18,1))


J_hist = np.zeros((1,iter_num))
var = np.zeros((1,iter_num))
for i in range(iter_num): 
    var[0,i] = i

for i in range(iter_num):
    
    correction = X_train.transpose().dot((X_train.dot(theta) - Y_train)) + lamda*theta
    theta = theta - alpha*correction   
    J_hist[0,i] = cost(X_train, Y_train, theta, lamda)/108

p = X_test.dot(theta)

Results = np.zeros((27,2))

for i in range(27):
    Results[i,0] = Y_test[i]
    Results[i,1] = p[i]
    
print ("First Shuffle:\n\n")

plt.plot(var[0],J_hist[0])
plt.show()

print ("Parameters: \n", theta)

print ("\nActual Values   Predicted Values \n",Results)

print ("\nTrain Cost: ", float(cost(X_train,Y_train,theta,lamda)/108))

print ("Test Cost: ", float(cost(X_test,Y_test,theta,lamda)/27))


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

theta = np.matlib.zeros((18,1))

for i in range(iter_num):
    
    correction = X_train.transpose().dot((X_train.dot(theta) - Y_train)) + lamda*theta
    theta = theta - alpha*correction
    J_hist[0,i] = cost(X_train, Y_train, theta, lamda)/108


p = X_test.dot(theta)

Results = np.zeros((27,2))

for i in range(27):
    Results[i,0] = Y_test[i]
    Results[i,1] = p[i]
    
print ("\n\nSecond Shuffle:\n\n")

plt.plot(var[0],J_hist[0])
plt.show()

print ("Parameters: \n", theta)

print ("\nActual Values   Predicted Values \n",Results)

print ("\nTrain Cost: ", float(cost(X_train,Y_train,theta,lamda)/108))

print ("Test Cost: ", float(cost(X_test,Y_test,theta,lamda)/27))

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

theta = np.matlib.zeros((18,1))

for i in range(iter_num):
    
    correction = X_train.transpose().dot((X_train.dot(theta) - Y_train)) + lamda*theta
    theta = theta - alpha*correction
    J_hist[0,i] = cost(X_train, Y_train, theta, lamda)/108


p = X_test.dot(theta)

Results = np.zeros((27,2))

for i in range(27):
    Results[i,0] = Y_test[i]
    Results[i,1] = p[i]
    
print ("\n\nThird Shuffle:\n\n")

plt.plot(var[0],J_hist[0])
plt.show()

print ("Parameters: \n", theta)

print ("\nActual Values   Predicted Values \n",Results)

print ("\nTrain Cost: ", float(cost(X_train,Y_train,theta,lamda)/108))

print ("Test Cost: ", float(cost(X_test,Y_test,theta,lamda)/27))

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

theta = np.matlib.zeros((18,1))

for i in range(iter_num):
    
    correction = X_train.transpose().dot((X_train.dot(theta) - Y_train)) + lamda*theta
    theta = theta - alpha*correction    
    J_hist[0,i] = cost(X_train, Y_train, theta, lamda)/108


p = X_test.dot(theta)

Results = np.zeros((27,2))

for i in range(27):
    Results[i,0] = Y_test[i]
    Results[i,1] = p[i]
    
print ("\n\nFourth Shuffle:\n\n")

plt.plot(var[0],J_hist[0])
plt.show()

print ("Parameters: \n", theta)

print ("\nActual Values   Predicted Values \n",Results)

print ("\nTrain Cost: ", float(cost(X_train,Y_train,theta,lamda)/108))

print ("Test Cost: ", float(cost(X_test,Y_test,theta,lamda)/27))


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

theta = np.matlib.zeros((18,1))

for i in range(iter_num):
    
    correction = X_train.transpose().dot((X_train.dot(theta) - Y_train)) + lamda*theta
    theta = theta - alpha*correction    
    J_hist[0,i] = cost(X_train, Y_train, theta, lamda)/108


p = X_test.dot(theta)

Results = np.zeros((27,2))

for i in range(27):
    Results[i,0] = Y_test[i]
    Results[i,1] = p[i]
    
print ("\n\nFifth Shuffle:\n\n")

plt.plot(var[0],J_hist[0])
plt.show()

print ("Parameters: \n", theta)

print ("\nActual Values   Predicted Values \n",Results)

print ("\nTrain Cost: ", float(cost(X_train,Y_train,theta,lamda)/108))

print ("Test Cost: ", float(cost(X_test,Y_test,theta,lamda)/27))
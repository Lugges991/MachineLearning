################################################################################
#implementing stochastic gradient descent with a simple
# learning schedule using numpy
#Author: Lucas Mahler
#GitHub: @Lugges991
################################################################################
import numpy as np
import matplotlib.pyplot as plt

#generate random, linear looking data
X = 3* np.random.rand(100,1)
#adding gaussian noise to the graph
# y = 6 + 2x + gaussian noise
y = 6 +2 * X + np.random.randn(100,1)

#add x0 = 1 to each instance using numpy's concatenation function
X_b = np.c_[np.ones((100,1)),X]

#number of data points
m = 100

#define number of learning cycles, called epochs
n_epochs = 50

#define learning hyperparameters
t0, t1 = 5,50

#define a learning schedule function
def learning_schedule(t):
    return t0 / (t + t1)

#randomly initialize theta
theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        #choose a random index in range of m
        random_index = np.random.randint(m)

        #determining x and y values of the current index
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]

        #calculate the gradient, eta and theta
        gradients = 2 * xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

print("stochastic theta: \n")
print(theta)

#new x values we want to predict y values for
X_new = np.array([[0],[2]])

#add x0 = 1 to each instance using numpy's concatenation function
X_new_b = np.c_[np.ones((2,1)), X_new]

#predict y for the new X using our model theta
y_hat = X_new_b.dot(theta)

print("\n y_hat: \n")
print(y_hat)

#plotting the values
plt.plot(X, y, 'b.')
plt.plot(X_new, y_hat, 'r-')
plt.axis([0,2,0,15])
plt.show()

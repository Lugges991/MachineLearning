################################################################################
#implementing batch gradient descent using numpy
#Author: Lucas Mahler
#GitHub: @Lugges991
################################################################################
import numpy as np
import matplotlib.pyplot as plt

#generate random, lienear looking data
X = 3* np.random.rand(100,1)
#adding gaussian noise to the graph
# y = 6 + 2x + gaussian noise
y = 6 +2 * X + np.random.randn(100,1)

#add x0 = 1 to each instance using numpy's concatenation function
X_b = np.c_[np.ones((100,1)),X]

#specify the lerning rate eta
eta = 0.1

#specify the number of iterations
n_iterations = 1000

#number of data points
m = 100

#randomly initialize theta
theta = np.random.randn(2,1)

#iterating through all datapoints Xi
for iteration in range(n_iterations):

    #calculate the gradient vector according to the formula:
    # gradient = 2 / m * transpose(X) * ( X * theta - y)
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta)-y)

    #calculate theta for the next step
    theta = theta - eta * gradients
print("batch theta: \n")
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

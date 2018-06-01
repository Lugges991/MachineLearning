################################################################################
#implementing linear regression using numpy
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

#find value of theta minimizing the cost function
#--> Normal equation
#--> theta_best = ((transpose(X) * X)^-1 ) * transpose(X) * y

#add x0 = 1 to each instance using numpy's concatenation function
X_b = np.c_[np.ones((100,1)),X]

#calculate theta_best using numpy's inverse and dot product function
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

#according to our initial y, theta(0) should be 6 and theta(1)
#should be 2
print(theta_best)

#due to the added noise, we came close but not 100% accurate
#now we make predictions using theta_best

#new x values we want to predict y values for
X_new = np.array([[0],[2]])

#add x0 = 1 to each instance using numpy's concatenation function
X_new_b = np.c_[np.ones((2,1)), X_new]

#predict y for the new X using our model theta
y_hat = X_new_b.dot(theta_best)

print(y_hat)

#plotting the values
plt.plot(X, y, 'b.')
plt.plot(X_new, y_hat, 'r-')
plt.axis([0,2,0,15])
plt.show()

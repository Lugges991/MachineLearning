################################################################################
#implementing linear regression using numpy
#Author: Lucas Mahler
#GitHub: @Lugges991
################################################################################

import numpy as np
import matplotlib.pyplot as plt

#number of data points
m = 100

#generate random, polynomial data
X = 6* np.random.rand(m,1) - 3

#adding gaussian noise to the graph
# y = 0.5x² + x + 2+ gaussian noise
y = 0.5 * X**2 + X + 2+ np.random.randn(m,1)

#add x0 = 1 to each instance using numpy's concatenation function
X_b = np.c_[np.ones((m,1)),X]

#adding the square of each feature as a new feature
X_bs = np.append(X_b, X**2, axis=1)

#calculate theta using numpys inverse and dot product function
theta_best = np.linalg.inv(X_bs.T.dot(X_bs)).dot(X_bs.T).dot(y)

print("\n theta_best: \n")
print(theta_best)

#thus our model is very close to the original


X_new = np.linspace(-3, 3, 100).reshape(100, 1)

X_new_b = np.c_[np.ones((m,1)),X_new]

X_new_bs =  np.append(X_new_b, X_new**2, axis=1)

y_hat = X_new_bs.dot(theta_best)

plt.plot(X, y, "b.")
plt.plot(X_new, y_hat, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])
plt.show()

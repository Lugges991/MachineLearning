################################################################################
#implementing polynomial regression using numpy and sklearn
#Author: Lucas Mahler
#GitHub: @Lugges991
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#number of data points
m = 100

#generate random, polynomial data
X = 6* np.random.rand(m,1) - 3

#adding gaussian noise to the graph
# y = 0.5x² + x + 2+ gaussian noise
y = 0.5 * X**2 + X + 2+ np.random.randn(m,1)

polynomial_features = PolynomialFeatures(degree=2, include_bias=False)

X_polynomial = polynomial_features.fit_transform(X)

linear_regression = LinearRegression()
linear_regression.fit(X_polynomial,y)

print("\n ß0, ß1, ß2: \n")
print(linear_regression.intercept_, linear_regression.coef_)

X_new=np.linspace(-3, 3, 100).reshape(100, 1)

X_new_polynomial = polynomial_features.transform(X_new)

y_new = linear_regression.predict(X_new_polynomial)

plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])
plt.show()

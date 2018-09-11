###############################################################################
# implementing linear regression using plain tensorflow
# Author: Lucas Mahler
# GitHub: @Lugges991
###############################################################################
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# get the housing data from the sklearn datasets
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# creating our feature and label variables
# by passing the numpy array representation as values
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# transposing X
XT = tf.transpose(X)

# using normal equation to compute theta
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

# starting the computation graph to first initialize all variables and then
# evaluate the operation thea_val
with tf.Session() as sess:
    theta_value = theta.eval()
print(theta_value)

# comparing our own model created with tf with numpy
X = housing_data_plus_bias
y = housing.target.reshape(-1, 1)
theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta_numpy)

# comparing with sklearn
lin_reg = LinearRegression()
lin_reg.fit(housing.data, housing.target.reshape(-1, 1))
print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])

print("tf == np: ", theta_value == theta_numpy)
print("tf == sklearn: ", theta_value == np.r_[
      lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])

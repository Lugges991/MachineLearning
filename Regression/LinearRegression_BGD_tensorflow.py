###############################################################################
# implementing linear regression with batch gradient descent
# Author: Lucas Mahler
# GitHub: @Lugges991
###############################################################################
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# get the housing data from the sklearn datasets
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# scaling the input features
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# manually computing the gradients
# defining hyperparams
n_epochs = 1000
learning_rate = 0.01
# creating our feature and label variables
# by passing the numpy array representation as values
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

# random initialization of theta
theta = tf.Variable(tf.random_uniform(
    [n + 1, 1], -1.0, 1.0, seed=42), name="theta")

# prediction = X * theta
y_pred = tf.matmul(X, theta, name="predictions")

# computing the mean squared error
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# computing the gradients accordign to the BGD formula
gradients = 2 / m * tf.matmul(tf.transpose(X), error)

# define the training operation of BGD
training_op = tf.assign(theta, theta - learning_rate * gradients)

# create a variable initializer
init = tf.global_variables_initializer()

# start a tf session
with tf.Session() as sess:
    # initialize all variables
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())

        # run the training operation for each epoch
        sess.run(training_op)

    # evaluate the errors
    best_theta = theta.eval()
print(best_theta)
sess.close()

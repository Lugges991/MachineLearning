################################################################################
#implementing best fit slope with y intercept (Linear Regression)
#Author: Lucas Mahler
#GitHub: @Lugges991
################################################################################
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from matplotlib import style
style.use('ggplot')

#initialize data arrays
X = np.array([1,2,3,4,5], dtype=np.float64)
y = np.array([5,4,6,5,6], dtype=np.float64)

#creating a best fit slope function returning the slope m
#and the y-intercept b
def bestFitSlopeIntercept(X,y):
    #calculating m according to the formula:
    # m = sumOf(Xi - mean(X))*(yi - mean(y)) / sumOf(Xi - mean(X))²
    m = (((mean(X) * mean(y)) - mean(X * y)) /
         ((mean(X)*mean(X)) - mean(X*X)))

    #calculate the y-intercept according to:
    # b = mean(y) - m * mean(X)
    b = mean(y) - m * mean(X)
    return m, b

#calculate the slope and y-intercept for our data
m ,b= bestFitSlopeIntercept(X,y)
print(m,b)

#initialize an array to hold our new y values and calculate them
#at the same tim
regression_line = [(m * Xi) + b for Xi in X]

#predict a y-value for the initially not contained Xi=7
X_pred = 7
yHat = (m * X_pred) + b

plt.scatter(X,y,color='#003F72',label='data')
plt.plot(X, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()

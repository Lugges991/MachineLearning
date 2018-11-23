################################################################################
# convolution operation implemented in python using numpy
#
# Author: Lucas Mahler
# GitHub: @Lugges991
################################################################################

import numpy as np
import matplotlib.pyplot as plt


def convolution_operation_rgb(image, kernel):
    output = np.zeros(image.shape)


    for l in range(image.shape[2]):
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                output[i,j,l] = (kernel * image[i:i+kernel.shape[0], j:j+kernel.shape[0]]).sum()





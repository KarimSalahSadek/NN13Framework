#ALL FUNCTIONS TAKE NUMPY ARRAYS AS INPUT AND RETURN NUMPY ARRAYS!!!!
#Numpy will automatically understand that 1x1 arrays are not arrays and return them correctly
#If grad is set to True , return the derivative instead

import numpy as np
import nn13framework.neural_networks.activation_functions as AF

def mean_square_loss(prediction ,label,grad=False):
    assert (label.shape == prediction.shape)
    # the function output
    col_len,row_len=label.shape
    if grad == False:
        return np.sum(((np.multiply((label-prediction),(label-prediction)))/(2*row_len)))
    # the function derivative
    elif grad == True:
        return -((label-prediction)/row_len)



def multinomial_loss(prediction ,label,grad=False):
    assert (label.shape == prediction.shape)
    # the function output
    if grad == False:
        return -np.sum((np.multiply(np.log(AF.softmax(prediction,False)),label )))
    # the function derivative
    elif grad == True:
        # return the derivative for each class by d itself
        return AF.softmax(prediction,False)-label


#ALL FUNCTIONS TAKE NUMPY ARRAYS AS INPUT AND RETURN NUMPY ARRAYS!!!!
#Numpy will automatically understand that 1x1 arrays are not arrays and return them correctly
#All Funcitons are static methods that don't require an instance to call

import numpy as np
import activation_functions as AF
@staticmethod
def mean_square_loss(prediction ,label,grad=False):
    # the function output
    if grad == False:
        return sum((label-prediction)*(label-prediction)/(2*len(label)))
    # the function derivative
    elif grad == True:
        return -(label-prediction/len(label))


@staticmethod
def multinomial_loss(prediction ,label,grad=False):
    # the function output
    if grad == False:
        return -sum(label*np.log(AF.softmax(prediction,False)) )
    # the function derivative
    elif grad == True:
        # return the derivative for each class by d itself
        return AF.softmax(prediction,False)-label


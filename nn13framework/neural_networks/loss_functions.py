#ALL FUNCTIONS TAKE NUMPY ARRAYS AS INPUT AND RETURN NUMPY ARRAYS!!!!
#Numpy will automatically understand that 1x1 arrays are not arrays and return them correctly
#If grad is set to True , return the derivative instead

import numpy as np
import activation_functions as AF

def mean_square_loss(prediction ,label,grad=False):
    assert (label.shape == prediction.shape)
    # transpose the matrices
    label=np.transpose(label)
    prediction=np.transpose(prediction)
    # the function output
    col_len,row_len=label.shape
    if grad == False:
        return np.sum(((np.multiply((label-prediction),(label-prediction)))/(2*row_len)))
    # the function derivative
    elif grad == True:
        return -np.transpose(((label-prediction)/row_len))



def multinomial_loss(prediction ,label,grad=False):
    assert (label.shape == prediction.shape)

    # the function output
    if grad == False:
        return -np.sum(np.multiply(np.log(prediction+10e-6),label ))
    # the function derivative
    elif grad == True:
        # return the derivative for each class by d itself
        no_of_examples, no_of_classes = prediction.shape
        delta = np.zeros((no_of_examples, no_of_classes))
        for j in range(no_of_examples):
            for i in range(no_of_classes):
                if  label[j][i]==1:
                    delta[j][i] = -(1 - prediction[j][i])
                else:
                    delta[j][i] = prediction[j][i]
        return delta

        



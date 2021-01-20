#ALL FUNCTIONS TAKE NUMPY ARRAYS AS INPUT AND RETURN NUMPY ARRAYS!!!!
#Numpy will automatically understand that 1x1 arrays are not arrays and return them correctly
#If grad is set to True , return the derivative instead

import numpy as np
import nn13framework.neural_networks.activation_functions as AF

#parent class
class loss_functions:
    pass

class mean_square_loss(loss_functions):
    # general attributes
    loss_derivative = None
    #init functions that take the model
    def __init__(self,model=None):
        self.model=model
        self.loss_derivative = None
    #calculate the function and the derivative but only return the function
    def evaluate(self,prediction,label):
        assert (label.shape == prediction.shape)
        # transpose the matrices
        label = np.transpose(label)
        prediction = np.transpose(prediction)
        # the function derivative
        self.loss_derivative = -np.transpose(((label - prediction) / row_len))
        # the function output
        col_len, row_len = label.shape
        return np.sum(((np.multiply((label - prediction), (label - prediction))) / (2 * row_len)))

    #return the previously calculated derivative
    def backward(self):
        return self.loss_derivative


class multinomial_loss(loss_functions):
    # general attributes
    loss_derivative = None

    # init functions that take the model
    def __init__(self, model=None):
        self.model = model
        self.loss_derivative = None

    # calculate the function and the derivative but only return the function
    def evaluate(self, prediction, label):
        assert (label.shape == prediction.shape)
        # the function derivative
        # return the derivative for each class by d all other classes
        no_of_examples, no_of_classes = prediction.shape
        delta = np.zeros((no_of_examples, no_of_classes))
        for j in range(no_of_examples):
            for i in range(no_of_classes):
                if label[j][i] == 1:
                    delta[j][i] = -(1 - prediction[j][i])
                else:
                    delta[j][i] = prediction[j][i]
        self.loss_derivative =  delta

        # the function output
        return -np.sum(np.multiply(np.log(prediction + np.exp(-15)), label))

    # return the previously calculated derivative
    def backward(self):
        return self.loss_derivative



#ALL FUNCTIONS TAKE NUMPY ARRAYS AS INPUT AND RETURN NUMPY ARRAYS!!!!
#Numpy will automatically understand that 1x1 arrays are not arrays and return them correctly
#If grad is set to True , return the derivative instead

import numpy as np
import nn13framework.neural_networks.activation_functions as AF

#parent class
class loss_functions:
    pass

class mean_square_loss(loss_functions):

    def __init__(self,model=None):
        self.model=model
        self.loss_derivative = None
    #calculate the function and the derivative but only return the function
    def evaluate(self,prediction,label):
        '''

        :param prediction: ,    :param label:
        each input is a kXd matrix where k is the number of examples and d is the number of input dimension
        :return: the sum of 0.5*(y_hat - y)^2 in forward propagation
        '''
        assert (label.shape == prediction.shape)
        # transpose the matrices
        label = np.transpose(label)
        prediction = np.transpose(prediction)
        # the function derivative
        col_len, row_len = label.shape
        self.loss_derivative = -np.transpose(((label - prediction) / row_len))
        # the function output
        return np.sum(((np.multiply((label - prediction), (label - prediction))) / (2 * row_len)))

    #return the previously calculated derivative
    def backward(self):
        '''

        :return: the derivative of the mean square loss = (y_hat - y)*dy_hat/dy_hat
        '''
        return self.loss_derivative


class multinomial_loss(loss_functions):
    
    # init functions that take the model
    def __init__(self, model=None):
        self.model = model
        self.loss_derivative = None

    # calculate the function and the derivative but only return the function
    def evaluate(self, prediction, label):
        '''

        :param prediction:
        :param label:
        each input is a kXd matrix where k is the number of examples and d is the number of input classes
        :return: the loss at the node at which the label is 1 = e^(node output)/sum of e^(all nodes output) in forward propagation
        '''
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
        return -np.sum(np.multiply(np.log(prediction + np.exp(-150)), label))

    # return the previously calculated derivative
    def backward(self):
        '''

        :return:the derivative of the mutinomial loss in backward propagation
        '''
        return self.loss_derivative


"""
     loss function for logistic regression problem & sigmoid activation function 
     loss = log(abs(Y/2 -0.5 + y_hat))
     incase--> Y = 1   loss = log(y_hat)       drevative =  1 / y_hat  
     incase--> Y = -1  loss = log(1-y_hat)     drevative = -1 / (1 - y_hat)
     parameters are 2 matrices : prediction is the output of the last layer & labels presents the actual output{1 or -1}
"""
class log_likehood_loss(loss_functions):
    # general attributes
    loss_derivative = None
    #init functions that take the model
    def __init__(self,model=None):
        self.model=model
        self.loss_derivative = None
    #calculate the function and the derivative but only return the function
    def evaluate(self,prediction,label):
        assert (label.shape == prediction.shape)
        # the function output
        L1 = (label > 0)
        L2 = (label <= 0)
        pred1 = np.multiply(prediction,L1)
        pred2 = np.multiply(prediction,L2)
        Loss = np.where(L2==0,np.log(pred1+np.exp(-15)),0) + np.where(L1==0,np.log(1-pred2+np.exp(-15)),0)

        # the function derivative
        dev = np.where(L2==0,1 / (pred1 + np.exp(-15)),0) + np.where(L1==0,-1 / (1 - pred2 + np.exp(-15)),0)
        self.loss_derivative = dev
        return -1 * np.sum(Loss)

    #return the previously calculated derivative
    def backward(self):
        return self.loss_derivative


"""
    loss function for logistic regression problem & linear activation function 
    loss = log(1 + exp(- Y *y_hat))
    derivative = 
    parameters are 2 matrices : 
    prediction is the output of the last layer each row represents 1 ex and each column represents certain node's output  
    labels presents the actual output
"""
class log_likehood2_loss(loss_functions):
    # general attributes
    loss_derivative = None
    #init functions that take the model
    def __init__(self,model=None):
        self.model=model
        self.loss_derivative = None
    #calculate the function and the derivative but only return the function
    def evaluate(self,prediction,label):
        assert (label.shape == prediction.shape)
        # the function output
        lap = -1 * label
        prod = np.multiply(lap,prediction)
        Loss = np.log(1+np.exp(prod))
        # print(Loss)
        # the function derivative
        dev = (1 * lap * np.exp(prod))/(1 + np.exp(prod))
        # print(dev)
        self.loss_derivative = dev
        return np.sum(Loss)

"""
    loss function for Binary classifier problem & linear activation function 
    loss = Max(0,-Y*Y_hat)
    derivative = -Y   if  -Y*Y_hat > 0 
    zero other wise
    parameters are 2 matrices : 
    prediction is the output of the last layer each row represents 1 ex and each column represents certain node's output for all exp 
    labels presents the actual output [1 , -1] or any 2 positive & negative values
"""
class Hing_loss(loss_functions):
    # general attributes
    loss_derivative = None
    #init functions that take the model
    def __init__(self,model=None):
        self.model=model
        self.loss_derivative = None
    #calculate the function and the derivative but only return the function
    def evaluate(self,prediction,label):
        assert (label.shape == prediction.shape)
        # the function output
        lap = -1 * label
        prod = np.multiply(lap,prediction)
        Loss = np.maximum(prod,0)
        # print(Loss)
        # the function derivative
        dev = np.where(prod > 0,lap,0)
        # print(dev)
        self.loss_derivative = dev
        return np.sum(Loss)

    #return the previously calculated derivative
    def backward(self):
        return self.loss_derivative


"""
    # loss function for Binary classifier problem & linear activation function 
    # loss = Max(0,1-Y*Y_hat)
    # derivative = -Y   if  1-Y*Y_hat > 0 
    # zero other wise
    # parameters are 2 matrices : 
    # prediction is the output of the last layer each row represents 1 ex and each column represents certain node's output for all exp 
    # labels presents the actual output [1 , -1] or any 2 positive & negative values
"""
class svm_Hing_loss(loss_functions):
    # general attributes
    loss_derivative = None
    #init functions that take the model
    def __init__(self,model=None):
        self.model=model
        self.loss_derivative = None
    #calculate the function and the derivative but only return the function
    def evaluate(self,prediction,label):
        assert (label.shape == prediction.shape)
        # the function output
        lap = -1 * label
        prod = np.multiply(lap,prediction) + 1
        Loss = np.maximum(prod,0)
        # print(Loss)
        # the function derivative
        dev = np.where(prod > 0,lap,0)
        # print(dev)
        self.loss_derivative = dev
        return np.sum(Loss)

    #return the previously calculated derivative
    def backward(self):
        return self.loss_derivative







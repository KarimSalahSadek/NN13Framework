#ALL FUNCTIONS TAKE NUMPY ARRAYS AS INPUT AND RETURN NUMPY ARRAYS!!!!
#Numpy will automatically understand that 1x1 arrays are not arrays and return them correctly
#If grad is set to True , return the derivative instead

import numpy as np
import nn13framework.neural_networks.activation_functions as AF

#parent class
class loss_functions:
    pass

class mean_square_loss(loss_functions):

    def __init__(self,model,regularization_parameter = 0):
        self.model=model
        self.loss_derivative = None
        self.lamda = regularization_parameter
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
        #the regularization term
        w = self.model.weights[-1]
        w = np.multiply(self.lamda / 2, np.multiply(w, w))
        w = np.sum(w)
        # the function output
        return np.sum(((np.multiply((label - prediction), (label - prediction))) / (2 * row_len)))+w

    #return the previously calculated derivative
    def backward(self):
        '''

        :return: the derivative of the mean square loss = (y_hat - y)*dy_hat/dy_hat
        '''

        return self.loss_derivative

    

class log_likehood_loss(loss_functions):
    """
        loss function for logistic regression problem & sigmoid activation function 
        loss = log(abs(Y/2 -0.5 + y_hat))
        incase--> Y = 1   loss = log(y_hat)       drevative =  1 / y_hat  
        incase--> Y = -1  loss = log(1-y_hat)     drevative = -1 / (1 - y_hat)
        parameters are 2 matrices : prediction is the output of the last layer & labels presents the actual output{1 or -1}
    """
    # general attributes
    loss_derivative = None
    #init functions that take the model
    def __init__(self,model,regularization_parameter = 0):
        self.model=model
        self.loss_derivative = None
        self.lamda = regularization_parameter
    #calculate the function and the derivative but only return the function
    def evaluate(self,prediction,label):
        assert (label.shape == prediction.shape)
        # the function output
        L1 = (label > 0)
        L2 = (label <= 0)
        pred1 = np.multiply(prediction,L1)
        pred2 = np.multiply(prediction,L2)
        Loss = np.where(L2==0,np.log(pred1+np.exp(-15)),0) + np.where(L1==0,np.log(1-pred2+np.exp(-15)),0)
        # the regularization term
        w = self.model.weights[-1]
        w = np.multiply(self.lamda / 2, np.multiply(w, w))
        w = np.sum(w)
        # the function derivative
        dev = np.where(L2==0,1 / (pred1 + np.exp(-15)),0) + np.where(L1==0,-1 / (1 - pred2 + np.exp(-15)),0)
        self.loss_derivative = dev
        return (-1 * (np.sum(Loss)) + w)

    #return the previously calculated derivative
    def backward(self):
        return self.loss_derivative

class log_likehood_alt_loss(loss_functions):
    """
        loss function for logistic regression problem & linear activation function 
        loss = log(1 + exp(- Y *y_hat))
        derivative = (-Y* exp(- Y *y_hat)) / (1 + exp(- Y *y_hat)) 
        parameters are 2 matrices : 
        prediction is the output of the last layer each row represents 1 ex and each column represents certain node's output  
        labels presents the actual output
    """
    # general attributes
    loss_derivative = None
    #init functions that take the model
    def __init__(self,model,regularization_parameter = 0):
        self.model=model
        self.loss_derivative = None
        self.lamda = regularization_parameter
    #calculate the function and the derivative but only return the function 
    def evaluate(self,prediction,label):
        assert (label.shape == prediction.shape)
        # the function output
        lap = -1 * label
        prod = np.multiply(lap,prediction)
        Loss = np.log(1+np.exp(prod))
        # the regularization term
        w = self.model.weights[-1]
        w = np.multiply(self.lamda / 2, np.multiply(w, w))
        w = np.sum(w)
        # the function derivative
        dev = (1 * lap * np.exp(prod))/(1 + np.exp(prod))
        self.loss_derivative = dev
        return np.sum(Loss)+w
    #return the previously calculated derivative
    def backward(self):
        return self.loss_derivative


class hinge_loss(loss_functions):
    """
        loss function for Binary classifier problem & linear activation function 
        loss = Max(0,-Y*Y_hat)
        derivative = -Y   if  -Y*Y_hat > 0 
        zero other wise
        parameters are 2 matrices : 
        prediction is the output of the last layer each row represents 1 ex and each column represents certain node's output for all exp 
        labels presents the actual output [1 , -1] or any 2 positive & negative values
    """
    # general attributes
    loss_derivative = None
    #init functions that take the model
    def __init__(self,model,regularization_parameter = 0):
        self.model=model
        self.loss_derivative = None
        self.lamda = regularization_parameter
    #calculate the function and the derivative but only return the function
    def evaluate(self,prediction,label):
        assert (label.shape == prediction.shape)
        # the function output
        lap = -1 * label
        prod = np.multiply(lap,prediction)
        Loss = np.maximum(prod,0)
        # the regularization term
        w = self.model.weights[-1]
        w = np.multiply(self.lamda / 2, np.multiply(w, w))
        w = np.sum(w)
        # the function derivative
        dev = np.where(prod > 0,lap,0)
        self.loss_derivative = dev
        return np.sum(Loss)+w

    #return the previously calculated derivative
    def backward(self):
        return self.loss_derivative



class svm_hinge_loss(loss_functions):
    """
        loss function for Binary classifier problem & linear activation function 
        loss = Max(0,1-Y*Y_hat)
        derivative = -Y   if  1-Y*Y_hat > 0 
        zero other wise
        parameters are 2 matrices : 
        prediction is the output of the last layer each row represents 1 ex and each column represents certain node's output for all exp 
        labels presents the actual output [1 , -1] or any 2 positive & negative values
    """
    # general attributes
    loss_derivative = None
    #init functions that take the model
    def __init__(self,model,regularization_parameter = 0):
        self.model=model
        self.loss_derivative = None
        self.lamda = regularization_parameter
    #calculate the function and the derivative but only return the function
    def evaluate(self,prediction,label):
        assert (label.shape == prediction.shape)
        # the function output
        lap = -1 * label
        prod = np.multiply(lap,prediction) + 1
        Loss = np.maximum(prod,0)
        # the regularization term
        w = self.model.weights[-1]
        w = np.multiply(self.lamda / 2, np.multiply(w, w))
        w = np.sum(w)
        # the function derivative
        dev = np.where(prod > 0,lap,0)
        self.loss_derivative = dev
        return np.sum(Loss)+w

    #return the previously calculated derivative
    def backward(self):
        return self.loss_derivative

#############################################   MULticlass loss funtion #####################################
class multinomial_loss(loss_functions):
    
    # init functions that take the model
    def __init__(self,model,regularization_parameter = 0):
        self.model = model
        self.loss_derivative = None
        self.lamda = regularization_parameter

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
        # the regularization term
        w = self.model.weights[-1]
        w = np.multiply(self.lamda / 2, np.multiply(w, w))
        w = np.sum(w)
        # the function output
        return (-1* (np.sum(np.multiply(np.log(prediction + np.exp(-150)), label))) + w)

    # return the previously calculated derivative
    def backward(self):
        '''

                :return:the derivative of the mutinomial loss in backward propagation
                '''
        return self.loss_derivative

  

class perceptron_criterion_loss(loss_functions):
    """
        perceptron criterion
        loss function for Multiclass problem & linear activation function 
        loss_i = Max(0, Y_i - Y_corrent)
        Loss = max loss_i
        dervative out = matrix whose size = prediction's size
        if Y_correct is not the maximum value of one ex. (one row) of prediction matrix derivative = -1 @ Y_correct 
        & derivative = 1 @ maximum value of one exp & rest of the dervatives = 0
        else if Y_correct is the maximum value --> then all the row = 0,0,.... in derivative Matrix for this ex 
        parameters are 2 matrices (for the first fun): 
        prediction is the output of the last layer each row represents 1 ex and each column represents certain node's output for all exp 
        labels presents Matrix (one hup) , same size as prediction
        2 returns : 
        first function :(evalute) returns one value (loss) & compute drivative 
        second function : (back word) return the drivativre (Matrix consist of  1 , -1 & zeros)
    """
    # general attributes
    loss_derivative = None
    #init functions that take the model
    def __init__(self,model,regularization_parameter = 0):
        self.model=model
        self.loss_derivative = None
        self.lamda = regularization_parameter
    #calculate the function and the derivative but only return the function
    def evaluate(self,prediction,label):
        assert(label.shape == prediction.shape)

        # the function output
        row_len,col_len  = label.shape
        lap = np.zeros(row_len) 
        loss = np.zeros(row_len)
        #Get the label value for each exp
        lap = np.sum(np.multiply(prediction, label),axis=1)
        #convert it into matrix each row 's size = prediction 's row 
        lap = np.transpose(np.tile(lap,(col_len,1)))

        #compute loss: loss is a vector of n elemnt where n = no, of ex.
        # & each element is the highest value in 1 ex  
        loss = np.max(np.maximum(prediction-lap,0),axis=1)

        #convert loss to matrix the same way like lap
        li = np.transpose(np.tile(loss,(col_len,1)))

        # the function derivative
        #return matrix : has the same size as pred
        # consist of 1 , -1 & 0,    
        ret_mat = np.zeros_like(prediction)
        #set -1 @ the right class if its out was not the highest for z certain ex
        ret_mat = np.where(((label == 1) & ((lap + li) != prediction)),-1,0) 
        # set 1 @ the node who has the highet output for a certin ex
        ret_mat = np.where(((lap + li) == prediction) & (label != 1) ,1,ret_mat)
        self.loss_derivative = ret_mat
        # the regularization term
        w = self.model.weights[-1]
        w = np.multiply(self.lamda / 2, np.multiply(w, w))
        w = np.sum(w)
        return np.sum(loss)+w

    #return the previously calculated derivative
    def backward(self):
        return self.loss_derivative


    """
        SVM 
        loss function for Multiclass problem & linear activation function 
        loss_i = Max(0, Y_i - Y_corrent + 1)
        dervative out = matrix whose size = prediction's size
        if Y_correct is not the  maximum value of one ex. (one row) of prediction matrix derivative = -1 * n @ Y_correct
        where n is the no of nodes that have higher values in one ex of (one row of)prediction Matrix than (Y_correct -1 ) for this e x  
        & derivative = 1 @  values higer than Y_correct  for tis ex & rest of the dervatives = 0
        else if Y_correct is the maximum value --> then all the row = 0,0,.... in derivative Matrix for this ex 
        parameters are 2 matrices (for the first fun): 
        prediction is the output of the last layer each row represents 1 ex and each column represents certain node's output for all exp 
        labels presents Matrix (one hup) , same size as prediction
        2 returns : 
        first function :(evalute) returns one value (loss) & compute drivative 
        second function : (back word) return the drivativre (Matrix consist of  1 , -1*n & zeros)
    """
class svm_multiclass_loss(loss_functions):
    # general attributes
    loss_derivative = None
    #init functions that take the model
    def __init__(self,model,regularization_parameter = 0):
        self.model=model
        self.loss_derivative = None
        self.lamda = regularization_parameter
    #calculate the function and the derivative but only return the function
    def evaluate(self,prediction,label):
        assert(label.shape == prediction.shape)
        # the function output
        row_len,col_len  = label.shape
        lap = np.zeros(row_len) 
        loss = np.zeros(row_len)
        #Get the label value for each exp
        lap = np.sum(np.multiply(prediction, label),axis=1)
        #convert it into matrix each row 's size = prediction 's row 
        lap = np.transpose(np.tile(lap,(col_len,1)))
        #compute loss: loss is a vector of n elemnt where n = no, of ex.
        # & each element is the highest value in 1 ex  
        loss = np.maximum(1 + prediction-lap,0)
        li = np.max(loss,axis=1)
        li = np.transpose(np.tile(li,(col_len,1)))
        #set loss to zero  @ the right label
        loss = np.where(label == 1, 0, loss)
        #compute the no. of nodes that has a higher output than the actual label
        no_max = np.sum(np.where(loss > 0,1,0),axis=1)
        no_max = np.transpose(np.tile(no_max,(col_len,1)))
    
        # the function derivative
        #return matrix : has the same size as pred
        # consist of 1 , -1 & 0,    
        ret_mat = np.zeros_like(prediction)
        #set -1 @ the right class if its out was not the highest for z certain ex
        ret_mat = np.where(((label == 1) & ((li + lap -1) != prediction)),-1*no_max,0) 
        # set 1 @ the node who has the highet output for a certin ex
        ret_mat = np.where(((prediction + 1) > lap ) & (label != 1) ,1,ret_mat)
        self.loss_derivative = ret_mat
        # the regularization term
        w = self.model.weights[-1]
        w = np.multiply(self.lamda / 2, np.multiply(w, w))
        w = np.sum(w)
        return ((np.sum(loss)/row_len)+w)


    #return the previously calculated derivative
    def backward(self):
        return self.loss_derivative



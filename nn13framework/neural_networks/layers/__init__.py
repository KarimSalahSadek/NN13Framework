from nn13framework.neural_networks import activation_functions
import numpy as np

#Layer class , a model mainly consists of multiple objects of this class
class layer:
    pass

#Network Layers
class linear(layer):
    
    def __init__(self,input_dim,output_dim, use_bias = True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = None
        self.weight = None
        self.use_bias = use_bias
        self.is_activation = False
        self.input = None
        
    def forward(self,data_in):
        """
            Data input must be a matrix of row vectors, each row vector is an example
        """
        data_in = np.array(data_in)
        if self.use_bias:
            data_out = np.dot(self.weight,np.hstack((data_in,np.ones((data_in.shape[0],1)))).T)
            self.input = np.hstack((data_in,np.ones((data_in.shape[0],1)))).T
        else:
            data_out = np.dot(self.weight,data_in.T)
            self.input = data_in

        return data_out

    def backward(self,grad_data_out):
        """
        Returns grad of weight , grad of input data
        """
        grad_w = np.dot(grad_data_out,self.input.T)
        grad_data_in = np.dot(self.weight.T,grad_data_out)
        return grad_w , grad_data_in

###########################################################################
#Activation Layers
class sigmoid(layer):
    
    def __init__(self):
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None
        self.input_dim = None
        self.output_dim = None
        
    def forward(self,data_in):
        self.last_input = data_in
        data_out = activation_functions.sigmoid(data_in)
        return data_out

    def backward(self,data_out):
        """
        Returns None , grad of input data
        """
        grad_data_in = activation_functions.sigmoid(self.last_input,grad=True)
        return None , grad_data_in
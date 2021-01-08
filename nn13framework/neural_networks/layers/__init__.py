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
        if self.layer_num == 0:
            if len(data_in.shape) >1:
                assert((data_in.shape[0]>1) or (data_in.shape[1]>1))
            data_in = np.reshape(data_in,(-1,1))
        if self.use_bias:
            data_out = np.dot(self.weight,np.vstack((data_in,1)))
            self.input = np.vstack((data_in,1))
        else:
            data_out = np.dot(self.weight,data_in)
            self.input = data_in

        return data_out

    def backward(self,grad_data_out):
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
        grad = activation_functions.sigmoid(self.last_input,grad=True)
        return grad
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
        if self.layer_num != 0:
            data_in = np.array(data_in).T
        if self.use_bias:
            data_out = np.dot(self.weight[:,:-1],data_in.T) + np.array([self.weight[:,-1]]).T #bias
        else:
            data_out = np.dot(self.weight,data_in.T)
        self.input = data_in

        return data_out

    def backward(self,grad_data_out):
        """
        Returns grad of weight , grad of input data
        """
        if self.use_bias:
            grad_w = np.hstack((np.dot(grad_data_out,self.input),np.sum(grad_data_out,axis=1,keepdims=True)))
            #grad w = dw + db = sigma (grad_out_i * in_i) + sigma (grad_out_i)
        else:
            grad_w = np.dot(grad_data_out,self.input)
        grad_data_in = np.dot(self.weight[:,:-1].T,grad_data_out)
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
        grad_z = activation_functions.sigmoid(self.last_input,grad=True)
        assert(grad_z.shape == data_out.shape)
        grad_data_in = np.multiply(grad_z,data_out)
        return None , grad_data_in

class relu(layer):
    
    def __init__(self):
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None
        self.input_dim = None
        self.output_dim = None
        
    def forward(self,data_in):
        self.last_input = data_in
        data_out = activation_functions.relu(data_in)
        return data_out

    def backward(self,data_out):
        """
        Returns None , grad of input data
        """
        grad_z = activation_functions.relu(self.last_input,grad=True)
        assert(grad_z.shape == data_out.shape)
        grad_data_in = np.multiply(grad_z,data_out)
        return None , grad_data_in

class softmax(layer):
    
    def __init__(self):
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None
        self.input_dim = None
        self.output_dim = None
        
    def forward(self,data_in):
        self.last_input = data_in
        
        data_out = activation_functions.softmax(data_in)
        return data_out

    def backward(self,data_out):
        """
        Returns None , grad of input data
        """
        grad_z = activation_functions.softmax(self.last_input,grad=True)
        assert(grad_z.shape == data_out.shape)
        grad_data_in = np.multiply(grad_z,data_out)
        return None , grad_data_in

from ...neural_networks import activation_functions

#Layer class , a model mainly consists of multiple objects of this class
class layer:
    pass

class linear(layer):
    
    
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = None
        self.weight = None
        
    def forward(self,data_in):
        
        return data_out
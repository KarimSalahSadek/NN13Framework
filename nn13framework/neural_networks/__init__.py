import numpy as np
import nn13framework.neural_networks.layers as layers

class model:

    #History dictionary is made to save training data NOT model parameters
    def __init__(self,name = None,date_created = None):
        self.name = name
        self.date_created = date_created
        self.weights = None #list of numpy array later
        self.history = {}
        self.history['true_positives'] = []
        self.history['true_negatives'] = []
        self.history['false_positives'] = []
        self.history['false_negatives'] = []
        self.history['accuracy'] = [0]
        self.history['validation_accuracy'] = [0]
        self.history['batch_loss'] = []
        self.history['epoch_loss'] = [None]
        self.history['validation_loss'] = [None]
        self.history['precision'] = []
        self.history['recall']= []
        self.history['f1score'] = []
        self.history['epoch_number'] = 0
        self.layers = [] #list of layers later
        self.evaluate_mode = False
        pass
    
    def set_evaluate_mode(self,mode):
        '''
            Sets the mode (boolean) of the model to either training mode or evaluation mode
            Only affects dropout layers
        '''
        assert(type(mode)==type(bool())),"mode parameter must be either True or False"
        self.evaluate_mode = mode
        for layer in self.layers:
            if layer.layer_name == 'dropout':
                layer.evaluate_mode = mode
    
    #Initializes self.weights , has severals modes
    def initialize_weights(self,mode,a, b = None):
        '''
        mode can take three values:
        1- 'const': in this case all weights are initialized to the constant value a, while b is discarded
        2- 'normal_dist' in this case all weights are initialized from a normal distribution
        with mean = a and std_deviation = b
        3- 'uniform_dist' in this case all weights are initialized from a uniform distribution
        with min = a and max = b
        '''
        if(mode == 'const'):
            for layer in self.layers:
                if layer.weight is None:
                    continue
                shape = tuple(layer.weight.shape)
                layer.weight = np.ones(shape) * a
        elif (mode == 'normal_dist'):
            assert(b is not None),'b parameter (std_deviation) must be provided for this mode'
            for layer in self.layers:
                if layer.weight is None:
                    continue
                shape = layer.weight.shape
                layer.weight = np.random.normal(a,b,shape)
        elif (mode == 'uniform_dist'):
            assert(b is not None),'b parameter (uniform_dist_max) must be provided for this mode'
            for layer in self.layers:
                if layer.weight is None:
                    continue
                shape = layer.weight.shape
                layer.weight = np.random.uniform(a,b,shape)


        self.weights = [layer.weight for layer in self.layers if layer.weight is not None]

    def forward(self,data_in):
        '''
            Data input must be a matrix of row vectors, each row vector is an example
            Returns the output after one forward propagation through the entire network
        '''
        x = data_in
        for layer in self.layers:
            x = layer.forward(x)
        return x.T

    def add_layer(self,layer):
        '''
            Adds a layer to the model, layer is an 'nn13framework.neural_network.layers.layer' child object
        '''
        assert(hasattr(layer,'layer_name')),"Cannot add object belonging to abstract class 'layer'"
        assert(isinstance(layer,type(layers.layer()))),"Input must be a layer object!"
        if self.layers is None:
            self.layers = []
        if layer.is_activation:
                assert(len(self.layers)>0),"Cannot have the first layer as an activation layer"
        if layer.layer_name == 'linear':
            first = True
            for searching_layer in self.layers:
                if searching_layer.layer_name=='linear':
                    first = False
                    break
            layer.first_linear = first
            new_weight = None
            if layer.use_bias:
                new_weight = np.random.uniform(0,1/layer.input_dim**0.5,(layer.output_dim,layer.input_dim+1)) # +1 for bias
            else:
                new_weight = np.random.uniform(0,1/layer.input_dim**0.5,(layer.output_dim,layer.input_dim))
            layer.weight = new_weight
        elif layer.layer_name == 'convolution':
            new_weight = np.random.randn(layer.n_filter, layer.d_X, layer.h_filter, layer.w_filter) / np.sqrt(layer.n_filter / 2.)
            layer.weight =  new_weight

        layer_num = len(self.layers)
        layer.layer_num = layer_num
        self.layers.append(layer)

        self.weights = [layer.weight for layer in self.layers if layer.weight is not None]

    def remove_layer(self,layer_number):
        '''
            Removes the layer with the given number, layer counts starts at 0
        '''
        assert(type(layer_number)==type(1)),'layer_number must be an integer'
        for i in range(layer_number,len(self.layers)-1,1):
            self.layers[i] = self.layers[i+1]
            self.layers[i].layer_num -= 1

        self.weights = [layer.weight for layer in self.layers if layer.weight is not None]

class LeNet_5(model):
    
    def __init__(self,date_created = None):
        super().__init__('LeNet_5',date_created)
        self.add_layer(layers.convolution(1,6,5,1,2))
        self.add_layer(layers.tanh())
        self.add_layer(layers.average_pool(2,2))
        self.add_layer(layers.convolution(6,16,5,1,0))
        self.add_layer(layers.average_pool(2,2))
        self.add_layer(layers.flatten())
        self.add_layer(layers.linear(400,120))
        self.add_layer(layers.tanh())
        self.add_layer(layers.linear(120,84))
        self.add_layer(layers.tanh())
        self.add_layer(layers.linear(84,10))
        self.add_layer(layers.softmax())
        

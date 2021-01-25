import numpy as np
from ..neural_networks import layers

class model:
    """
        A neural network model object which holds all the paramters and the architicture of the neural network
    """
    def __init__(self,name = None,date_created = None):
        '''
            name (optional): string that gives a name for the model
            date_created (optional): string to save the date_created, the user provides the correct date themselves
        '''
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
        self.history['epoch_number'] = 0
        self.layers = [] #list of layers later
        self.evaluate_mode = False
        pass
    
    def reset_history(self):
        '''
            Resets the history directory for the model
            The history directory is used to store information about the last training session
        '''
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
        self.history['epoch_number'] = 0

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
    
    def initialize_weights(self,mode,a = None , b = None):
        '''
            Initializes self.weights , has severals modes
            mode can take four values:
            1- 'const': in this case all weights are initialized to the constant value a, while b is discarded
            2- 'normal_dist' in this case all weights are initialized from a normal distribution
            with mean = a and std_deviation = b
            3- 'uniform_dist' in this case all weights are initialized from a uniform distribution
            with min = a and max = b
            4- 'xavier'  in this case all weights are initialized using the Xavier Initialization method
        '''
        if(mode == 'const'):
            assert(a is not None),'a parameter (const) must be provided for this mode'
            for layer in self.layers:
                if layer.weight is None:
                    continue
                shape = tuple(layer.weight.shape)
                layer.weight = np.ones(shape) * a
        elif (mode == 'normal_dist'):
            assert(a is not None),'a parameter (std_deviation) must be provided for this mode'
            assert(b is not None),'b parameter (std_deviation) must be provided for this mode'
            for layer in self.layers:
                if layer.weight is None:
                    continue
                shape = layer.weight.shape
                layer.weight = np.random.normal(a,b,shape)
        elif (mode == 'uniform_dist'):
            assert(a is not None),'a parameter (uniform_dist_max) must be provided for this mode'
            assert(b is not None),'b parameter (uniform_dist_max) must be provided for this mode'
            for layer in self.layers:
                if layer.weight is None:
                    continue
                shape = layer.weight.shape
                layer.weight = np.random.uniform(a,b,shape)
        elif (mode == 'xavier'):
            for layer in self.layers:
                if layer.weight is None:
                    continue
                shape = layer.weight.shape
                if layer.layer_name == 'convolution':
                    continue
                layer.weight = np.random.randn(shape[0],shape[1])*np.sqrt(2/layer.input_dim)


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
        if layer.layer_name == 'linear' or layer.layer_name == 'linear_svm_output':
            first = True
            for searching_layer in self.layers:
                if searching_layer.layer_name=='linear' or searching_layer.layer_name=='linear_svm_output' :
                    first = False
                    break
            layer.first_linear = first
            new_weight = None
            if layer.use_bias:
                new_weight = np.ones((layer.output_dim,layer.input_dim+1)) # +1 for bias
            else:
                new_weight = np.ones((layer.output_dim,layer.input_dim))
            self.initialize_weights('xavier',layer.input_dim)
            layer.weight = new_weight
        elif layer.layer_name == 'convolution':
            new_weight = np.random.randn(layer.n_filter, layer.d_X, layer.h_filter, layer.w_filter) * np.sqrt(2/layer.n_filter)
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
    '''
        LeNet_5 architecture
        Can be loaded directly as a model object
    '''
    def __init__(self,name = 'LeNet_5',date_created = None):
        super().__init__(name,date_created)
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
        

import numpy as np

class model:

    #History dictionary is made to save training data NOT model parameters
    def __init__(self,name = None,date_created = None):
        self.name = name
        self.date_created = date_created
        self.weights = None #list of numpy array later
        self.history = {}
        self.history['true_positives'] = 0
        self.history['true_negatives'] = 0
        self.history['false_positives'] = 0
        self.history['false_negatives'] = 0
        self.history['accuracy'] = 0
        self.history['validation_accuracy'] = 0
        self.history['loss'] = 0
        self.history['validation_loss'] = 0
        self.layers = None #list of layers later
        pass

    #Initializes self.weights , has severals modes
    def initialize_weights(self,mode = 'uniform_dist',a = 0 , b = 1):
        '''
        mode can take three values:
        1- 'const': in this case all weights are initialized to the constant value a, while b is discarded
        2- 'normal_dist' in this case all weights are initialized from a normal distribution
        with mean = a and std_deviation = b
        3- 'uniform_dist' in this case all weights are initialized from a uniform distribution
        with min = a and max = b
        '''
        
        if(mode == 'const'):
            for i in range(len(self.weights)):
                shape = tuple(self.weights[i].shape)
                self.weights[i] = np.ones(shape) * a
        elif (mode == 'normal_dist'):
            for i in range(len(self.weights)):
                shape = self.weights[i].shape
                self.weights[i] = np.random.normal(a,b,shape)
        elif (mode == 'uniform_dist'):
            for i in range(len(self.weights)):
                shape = self.weights[i].shape
                self.weights[i] = np.random.uniform(a,b,shape)


    # Adds a new layer
    def add_layer(self,layer):
        assert(isinstance(layer,layers.layer)),"Input must be a layer object!"
        if self.layers is None:
            self.layers = []
        else:
            assert(layer.input_dim == self.layers[-1].output_dim)
        if self.weights is None:
            self.weights = []
            layer_num = 0
            layer.layer_num = layer_num
            self.weights.append(np.random.uniform(-1,1,(layer.input_dim+1,layer.output_dim)))
            self.layers.append(layer)
            return
        layer_num = len(self.layers)
        layer.layer_num = layer_num
        self.weights.append(np.random.uniform(-1,1,(layer.input_dim+1,layer.output_dim))) # +1 for bias

        self.layers.append(layer)
    
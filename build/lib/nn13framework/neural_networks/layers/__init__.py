from nn13framework.neural_networks import activation_functions
import numpy as np

#Layer class , a model mainly consists of multiple objects of this class
class layer:
    pass

#Network Layers
class linear(layer):
    """
        Also called dense or fully-connected layer
        all input neurons are connected to all output neurons
        
        output = w_1 * input1 + w_2* input_2 + .... w_n*input_n
    """

    def __init__(self,input_dim,output_dim, use_bias = True):
        """
            input_dim: number of neurons in the previous layer
            output_dim: number of neurons in the layer
            use_bias: boolean value for whether or not a bias term is added
        """
        self.layer_name = 'linear'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = None
        self.weight = None
        self.use_bias = use_bias
        self.is_activation = False
        self.last_input = None
        self.first_linear = False
        
    def forward(self,data_in):
        """
            Data input must be a matrix of row vectors, each row vector is an example
        """
        data_in = np.array(data_in)
        if not self.first_linear:
            data_in = np.array(data_in).T
        if self.use_bias:
            data_out = np.dot(self.weight[:,:-1],data_in.T) + np.array([self.weight[:,-1]]).T #bias
        else:
            data_out = np.dot(self.weight,data_in.T)
        self.last_input = data_in

        return data_out

    def backward(self,grad_data_out):
        """
        Returns grad of weight , grad of input data
        """
        if self.use_bias:
            grad_w = np.hstack((np.dot(grad_data_out,self.last_input),np.sum(grad_data_out,axis=1,keepdims=True)))
            #grad w = dw + db = sigma (grad_out_i * in_i) + sigma (grad_out_i)
        else:
            grad_w = np.dot(grad_data_out,self.last_input)
        grad_data_in = np.dot(self.weight[:,:-1].T,grad_data_out)
        return grad_w , grad_data_in

class linear_svm_output(layer):
    """
        A modified version of the linear layer
        Only works with svm_multiclass_loss and preceptron_criterion
        This layer must always be the last linear layer in the network
    """
    def __init__(self,input_dim,output_dim, use_bias = True):
        """
            input_dim: number of neurons in the previous layer
            output_dim: number of neurons in the layer
            use_bias: boolean value for whether or not a bias term is added
        """
        self.layer_name = 'linear_svm_output'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = None
        self.weight = None
        self.use_bias = use_bias
        self.is_activation = False
        self.last_input = None
        self.first_linear = False
        
    def forward(self,data_in):
        """
            Data input must be a matrix of row vectors, each row vector is an example
        """
        data_in = np.array(data_in)
        if not self.first_linear:
            data_in = np.array(data_in).T
        if self.use_bias:
            data_out = np.dot(self.weight[:,:-1],data_in.T) + np.array([self.weight[:,-1]]).T #bias
        else:
            data_out = np.dot(self.weight,data_in.T)
        self.last_input = data_in

        return data_out

    def backward(self,grad_data_out):
        """
        Returns grad of weight , grad of input data
        """
        inp = self.last_input
        delta = grad_data_out.T
        
        no_of_exp1,no_of_nodes_per_layer = delta.shape
        no_of_exp2,no_of_weights_per_node = inp.shape
        assert(no_of_exp1==no_of_exp2)

        if (self.use_bias):
            ones =np.eye(no_of_exp1) 
            acc_bias = np.zeros((no_of_nodes_per_layer,no_of_exp1))
            tensor2 = np.einsum('ij,ik->ijk',ones,delta)

        acc_delta = np.zeros((no_of_nodes_per_layer,no_of_weights_per_node))
        tensor1 = np.einsum('ij,ik->ijk', inp,delta)
        for i in range(no_of_exp1):
            acc_delta += np.transpose(tensor1[i])
            if (self.use_bias):
                acc_bias += np.transpose(tensor2[i]) 
            
        acc_delta = acc_delta / no_of_exp1

        if (self.use_bias):
            # add bias as last column
            acc_bias =np.transpose(np.sum(acc_bias,axis=1)/no_of_exp1)
            acc_delta = np.c_[acc_delta,acc_bias]

        grad_w = acc_delta

        delta = grad_data_out.T
        weights = self.weight

        delta = np.transpose(delta)
        no_of_nodes_per_layer1,no_of_exp1, = delta.shape
        no_of_nodes_per_layer2,no_of_weights_per_node = weights.shape

        if (self.use_bias):
            no_of_weights_per_node = no_of_weights_per_node - 1
            weights = np.delete(weights,no_of_weights_per_node,axis=1)

        acc_delta = np.zeros((no_of_exp1,no_of_weights_per_node))
        tensor1 = np.einsum('ij,ik->ijk', weights,delta)
        for i in range(no_of_nodes_per_layer1):
            acc_delta += np.transpose(tensor1[i])
        acc_delta = acc_delta / no_of_nodes_per_layer1

        grad_data_in = acc_delta.T

        return grad_w , grad_data_in

class convolution(layer):
    """
    initalization fn:

    this fn is convolution inialization ,will take needed parameter for convolution (filter size,stride,padding
    ,depth_out(number of filter) , depth_in (depth of image)
    filter size must be integer or tuple of length 2
    """


    def __init__(self, depth_in , depth_out, filter_size, stride, padding):
        self.stride = stride
        self.last_input = None
        self.layer_name = 'convolution'
        self.layer_num = None
        self.is_activation = False


        self.n_filter =  depth_out

        self.h_filter, self.w_filter = None , None
        if type(filter_size) == type(int()):
             self.h_filter = self.w_filter = filter_size
        elif type(filter_size) == type(tuple()) and len(filter_size)==2:
            self.h_filter, self.w_filter = filter_size
        else:
            raise Exception("Invalid filter size! Must be int() or tuple() of length 2!")
        self.stride, self.padding = stride, padding

        self.weight = None
        self.d_X = depth_in
        self.h_out = None
        self.w_out = None


    def forward(self, X):
        """
        argument X: input images with width ,height, depth
        return:out: weight with dimensions (dimension of self_col,1)
        """

        self.last_input = X
        n_X = X.shape[0]
        d_X = X.shape[1]
        h_X = X.shape[2]
        w_X = X.shape[3]

        self.h_out = (h_X - self.h_filter + 2 * self.padding) // self.stride + 1
        self.w_out = (w_X - self.w_filter + 2 * self.padding) // self.stride + 1

        self.X_col = im2col_indices(X, self.h_filter, self.w_filter, stride=self.stride, padding=self.padding)
        W_row = self.weight.reshape(self.n_filter, -1)

        out = W_row @ self.X_col
        out = out.reshape(self.n_filter, self.h_out, self.w_out, n_X)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):

        """
        argument dout: derivative of out in forward fn
        return: dw,dx
        """
        dout_flat = dout.transpose(1, 2, 3, 0).reshape(self.n_filter, -1)
        dW = dout_flat @ self.X_col.T
        dW = dW.reshape(self.weight.shape)
        W_flat = self.weight.reshape(self.n_filter, -1)
        dX_col = W_flat.T @ dout_flat
        shape = self.last_input.shape
        dX = col2im_indices(dX_col, shape, self.h_filter,self.w_filter, self.padding, self.stride)
        return dW , dX                                     
                                    



###########################################################################
#Activation Layers
class sigmoid(layer):
    """
        A layer that utilizes this activation function
    """
    def __init__(self):
        self.layer_name = 'sigmoid'
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None
        
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
    """
        A layer that utilizes this activation function
    """    
    def __init__(self):
        self.layer_name = 'relu'
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None
        
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

class tanh(layer):
    """
        A layer that utilizes this activation function
    """    
    def __init__(self):
        self.layer_name = 'tanh'
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None
        
    def forward(self,data_in):
        self.last_input = data_in
        data_out = activation_functions.tanh(data_in)
        return data_out

    def backward(self,data_out):
        """
        Returns None , grad of input data
        """
        grad_z = activation_functions.tanh(self.last_input,grad=True)
        assert(grad_z.shape == data_out.shape)
        grad_data_in = np.multiply(grad_z,data_out)
        return None , grad_data_in

class hard_tanh(layer):
    """
        A layer that utilizes this activation function
    """    
    def __init__(self):
        self.layer_name = 'hard_tanh'
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None
        
    def forward(self,data_in):
        self.last_input = data_in
        data_out = activation_functions.hard_tanh(data_in)
        return data_out

    def backward(self,data_out):
        """
        Returns None , grad of input data
        """
        grad_z = activation_functions.hard_tanh(self.last_input,grad=True)
        assert(grad_z.shape == data_out.shape)
        grad_data_in = np.multiply(grad_z,data_out)
        return None , grad_data_in

class sign(layer):
    """
        A layer that utilizes this activation function
    """    
    def __init__(self):
        self.layer_name = 'sign'
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None
        
    def forward(self,data_in):
        self.last_input = data_in
        data_out = activation_functions.sign(data_in)
        return data_out

    def backward(self,data_out):
        """
        Returns None , grad of input data
        """
        grad_z = activation_functions.sign(self.last_input,grad=True)
        assert(grad_z.shape == data_out.shape)
        grad_data_in = np.multiply(grad_z,data_out)
        return None , grad_data_in

class leaky_relu(layer):
    """
        A layer that utilizes this activation function
    """    
    def __init__(self):
        self.layer_name = 'leaky_relu'
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None
        
    def forward(self,data_in):
        self.last_input = data_in
        data_out = activation_functions.leaky_relu(data_in)
        return data_out

    def backward(self,data_out):
        """
        Returns None , grad of input data
        """
        grad_z = activation_functions.leaky_relu(self.last_input,grad=True)
        assert(grad_z.shape == data_out.shape)
        grad_data_in = np.multiply(grad_z,data_out)
        return None , grad_data_in

class elu(layer):
    """
        A layer that utilizes this activation function
    """    
    def __init__(self):
        self.layer_name = 'elu'
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None
        
    def forward(self,data_in):
        self.last_input = data_in
        data_out = activation_functions.elu(data_in)
        return data_out

    def backward(self,data_out):
        """
        Returns None , grad of input data
        """
        grad_z = activation_functions.elu(self.last_input,grad=True)
        assert(grad_z.shape == data_out.shape)
        grad_data_in = np.multiply(grad_z,data_out)
        return None , grad_data_in


class softmax(layer):
    """
        A layer that utilizes this activation function
    """    
    def __init__(self):
        self.layer_name = 'softmax'
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None
        
    def forward(self,data_in):
        self.last_input = data_in
        
        data_out = activation_functions.softmax(data_in)
        return data_out

    def backward(self,data_out):
        """
        Returns None , grad of input data
        """
        grad_data_in = []
        d_out = data_out.T
        inp = self.last_input.T
        for i in range(d_out.shape[0]):
            grad_data_in.append(np.multiply(d_out,inp))
        grad_data_in = np.mean(grad_data_in,axis=0).T
        return None , grad_data_in

class dropout(layer):
    """
        Turns off neurons randomly, each neuron has probability (1- keep_probability) to be turned off
        Only works when the model is not working in evaluation mode
    """    
    def __init__(self,keep_probability):
        """
            keep_probability: constant fraction that determines the probability for keeping each neuron
        """
        self.layer_name = "dropout"
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None
        self.p = keep_probability
        self.prob_matrix = None
        self.evaluate_mode = False
        
    def forward(self,data_in):
        if self.evaluate_mode == False:
            self.prob_matrix = np.random.binomial(1,self.p,size=data_in.shape)
            prob = self.p
        elif self.evaluate_mode:
            self.prob_matrix = np.ones(data_in.shape)
            prob = 1
        data_out = np.multiply(data_in,self.prob_matrix)/prob
        return data_out

    def backward(self,data_out):
        """
        Returns None , grad of input data
        """
        grad_data_in = np.multiply(self.prob_matrix,data_out)
        return None , grad_data_in


class flatten(layer):
    """
        Flattens the input but takes into account the batch size 
    """
    def __init__(self):
        self.layer_name = "flatten"
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None
        self.input_size = None
        
    def forward(self,data_in):
        self.input_size = data_in.shape
        data_out = np.reshape(data_in,(data_in.shape[0],-1))
        return data_out

    def backward(self,data_out):
        grad_data_in = np.reshape(data_out,self.input_size)
        return None , grad_data_in


class max_pool(layer):
    """
            this is initialization fn for max pooling
            argument:filter size,stride
    """

    def __init__(self, filter_size, stride):
        self.layer_name = "max_pool"
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None

        self.size = filter_size
        self.stride = stride

        self.h_out = None
        self.w_out = None
        

    def forward(self, X):
        """
            this is forward fn for max pooling
            argument:x before max pooling 
            return: out(x after max pooling)
        """
        self.last_input = X
        n_X = X.shape[0]
        d_X = X.shape[1]
        h_X = X.shape[2]
        w_X = X.shape[3]

        self.h_out = (h_X - self.size) // self.stride + 1
        self.w_out = (w_X - self.size) // self.stride + 1

        X_reshaped = X.reshape(X.shape[0] * X.shape[1], 1, X.shape[2], X.shape[3])

        self.X_col = im2col_indices(X_reshaped, self.size, self.size, padding=0, stride=self.stride)

        self.max_indices = np.argmax(self.X_col, axis=0)
        out = self.X_col[self.max_indices, range(self.max_indices.size)]
        out = out.reshape(self.h_out, self.w_out, n_X,d_X).transpose(2, 3, 0, 1)
        return out

    

    def backward(self, dout):
        """
            this is backward fn in case of max pooling 
            argument:data_out (that is calculated from forward fn)
            return: None , dx

        """
        X = self.last_input
        n_X = X.shape[0]
        d_X = X.shape[1]
        h_X = X.shape[2]
        w_X = X.shape[3]

        dX_col = np.zeros_like(self.X_col)
        # flatten the gradient
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()

        dX_col[self.max_indices, range(self.max_indices.size)] = dout_flat

        # get the original X_reshaped structure from col2im
        shape = (n_X * d_X, 1, h_X, w_X)
        dX = col2im_indices(dX_col, shape, self.size,self.size, padding=0, stride=self.stride)
        dX = dX.reshape(n_X, d_X, h_X, w_X)
        return None , dX

class average_pool(layer):
    """
        this is initialization fn for average pooling
        argument:filter size,stride
    """

    def __init__(self, filter_size, stride):
        self.layer_name = "average_pool"
        self.layer_num = None
        self.weight = None
        self.is_activation = True
        self.last_input = None

        self.size = filter_size
        self.stride = stride

        self.h_out = None
        self.w_out = None

        

    def forward(self, X):
        """
            this is forward fn for average pooling
            argument:x before average pooling 
            return: out(x after average pooling)
        """
        self.last_input = X
        n_X = X.shape[0]
        d_X = X.shape[1]
        h_X = X.shape[2]
        w_X = X.shape[3]

        self.h_out = (h_X - self.size) // self.stride + 1
        self.w_out = (w_X - self.size) // self.stride + 1

        X_reshaped = X.reshape(X.shape[0] * X.shape[1], 1, X.shape[2], X.shape[3])

        self.X_col = im2col_indices(X_reshaped, self.size, self.size, padding=0, stride=self.stride)

        self.indices = np.argmax(self.X_col, axis=0)

        out = np.mean(self.X_col,axis=0)
        out = out.reshape(self.h_out, self.w_out, n_X,d_X).transpose(2, 3, 0, 1)
        return out

    

    def backward(self, dout):
        """
            this is backward fn in case of average pooling 
            argument:data_out (that is calculated from forward fn)
            return: None , dx

        """
        X = self.last_input
        n_X = X.shape[0]
        d_X = X.shape[1]
        h_X = X.shape[2]
        w_X = X.shape[3]

        dX_col = np.zeros_like(self.X_col)
        # flatten the gradient
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()

        dX_col[self.indices, range(self.indices.size)] = dout_flat

        # get the original X_reshaped structure from col2im
        shape = (n_X * d_X, 1, h_X, w_X)
        dX = col2im_indices(dX_col, shape, self.size,self.size, padding=0, stride=self.stride)
        dX = dX.reshape(n_X, d_X, h_X, w_X)
        return None , dX



def get_im2col_indices(x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """ 
        NOT FOR DIRECT USAGE
    """
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height,dtype='int32'), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height,dtype='int32'), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width,dtype='int32'), int(out_height))
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C,dtype='int32'), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height=3, field_width=3, padding=1, stride=1):
    """ 
        NOT FOR DIRECT USAGE
    """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,stride=1):
    """ 
        NOT FOR DIRECT USAGE
    """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

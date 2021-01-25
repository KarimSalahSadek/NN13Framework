import numpy as np

class optimizer:
    pass

class SGD(optimizer):
    """
    this is initialization fn for SGD
    argument: model, loss_fn, learning_rate, batch_size
    """
 
    def __init__(self, model,loss_fn, learning_rate):
        self.learning_rate = learning_rate
        self.batch_size = None
        self.model = model
        self.loss_fn= loss_fn
		

    def step(self):
        """
        this function update weight according to mini batch gradient descent
        """
        self.batch_size = self.model.layers[0].last_input.shape[0]
        grad_loss=self.loss_fn.backward()
        last_dx = grad_loss.T
        for i in reversed(range(len(self.model.layers))):
            dw, dx = self.model.layers[i].backward(last_dx)
            last_dx = dx
            # Update Rule
            if self.model.layers[i].is_activation:
                continue
            self.model.layers[i].weight -= (self.learning_rate * dw / self.batch_size )  - self.loss_fn.lamda*self.model.layers[i].weight*((i+1)==len(self.model.layers))

        self.model.weights = [layer.weight for layer in self.model.layers if layer.weight is not None]
	
	

class momuntum(optimizer):
    """
    this is initialization fn for momentum
    argument: model, loss_fn, learning_rate, beta
    """

    def __init__(self, model,loss_fn, learning_rate,beta=0.9):
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.beta = beta
        self.model = model
        self.v = []
        L = len(self.model.layers)
        for i in range(L):

            self.v.append(np.zeros_like(self.model.layers[i].weight))  # set v =0 in all layers






    def step(self):
        """
        this function update weight according to momentum
        """
        grad_loss=self.loss_fn.backward()
        last_dx = grad_loss.T
        for i in reversed(range(len(self.model.layers))):
            dw, dx = self.model.layers[i].backward(last_dx)
            last_dx = dx
            # Update Rule
            if self.model.layers[i].is_activation:
                continue
            self.v[i] = (self.beta * self.v[i]) - self.learning_rate * dw
            self.model.layers[i].weight += self.v[i]  - self.loss_fn.lamda*self.model.layers[i].weight*((i+1)==len(self.model.layers))

        self.model.weights = [layer.weight for layer in self.model.layers if layer.weight is not None]




class adagrad(optimizer):
    """
    this is initialization fn for adagrad
    argument: model, loss_fn, learning_rate, epsilon
    """

    def __init__(self, model,loss_fn, learning_rate,epsilon=1e-8):
        self.learning_rate = learning_rate
        self.model = model
        self.epsilon=epsilon
        self.loss_fn = loss_fn
        self.s = []
        L = len(self.model.layers)
        for i in range(L):

            self.s.append(np.zeros_like(self.model.layers[i].weight))
			
			
    def step(self):
        """
        this function update weight according to adagrad
        """
        grad_loss = self.loss_fn.backward()
        last_dx = grad_loss.T

        for i in reversed(range(len(self.model.layers))):
            dw, dx = self.model.layers[i].backward(last_dx)
            last_dx = dx

            if self.model.layers[i].is_activation:
                continue

            self.s[i] +=   np.power(dw, 2)#Ai accumelator
            self.model.layers[i].weight = self.model.layers[i].weight - (self.learning_rate * dw) / np.sqrt(self.s[i] + self.epsilon)  - self.loss_fn.lamda*self.model.layers[i].weight*((i+1)==len(self.model.layers))

        self.model.weights = [layer.weight for layer in self.model.layers if layer.weight is not None]
		
		
		




class RMSprop(optimizer):
    """
    this is initialization fn for RMSprop
    argument: model, loss_fn, learning_rate, epsilon
    """


    def __init__(self, model,loss_fn, learning_rate,epsilon=1e-8):
        self.learning_rate = learning_rate
        self.model = model
        self.epsilon=epsilon
        self.loss_fn = loss_fn
        self.s = []
        L = len(self.model.layers)
        for i in range(L):

            self.s.append(np.zeros_like(self.model.layers[i].weight))
			
		



    def step(self):
        """
        this function update weight according to RMS
        """
        grad_loss = self.loss_fn.backward()
        last_dx = grad_loss.T

        for i in reversed(range(len(self.model.layers))):
            dw, dx = self.model.layers[i].backward(last_dx)
            last_dx = dx

            if self.model.layers[i].is_activation:
                continue

            self.s[i] =(self.learning_rate *self.s[i]) + (1-self.learning_rate)*np.power(dw, 2)#Ai accumelator
            self.model.layers[i].weight = self.model.layers[i].weight - (self.learning_rate * dw) / np.sqrt(self.s[i] + self.epsilon)  - self.loss_fn.lamda*self.model.layers[i].weight*((i+1)==len(self.model.layers))

        self.model.weights = [layer.weight for layer in self.model.layers if layer.weight is not None]




class adadelta(optimizer):
    """
    this is initialization fn for adadelta
    argument: model, loss_fn, learning_rate, epsilon
    """

    def __init__(self, model, loss_fn, learning_rate, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = loss_fn
        self.s = []
        self.si = []
        L = len(self.model.layers)
        for i in range(L):
            self.s.append(np.ones_like(self.model.layers[i].weight))
            self.si.append(np.ones_like(self.model.layers[i].weight))

    def step(self):
        """
        this function update weight according to adadelta
        """
        grad_loss = self.loss_fn.backward()
        last_dx = grad_loss.T

        for i in reversed(range(len(self.model.layers))):
            dw, dx = self.model.layers[i].backward(last_dx)
            last_dx = dx

            if self.model.layers[i].is_activation:
                continue

            self.s[i] = self.learning_rate * (self.s[i]) + (1 - self.learning_rate) * dw ** 2  # Ai

            self.dwi = dw * ((np.sqrt(self.epsilon + self.si[i])) / (np.sqrt(self.epsilon + self.s[i])))  # delta wi

            self.model.layers[i].weight = self.model.layers[i].weight - (self.dwi) * self.learning_rate  - self.loss_fn.lamda*self.model.layers[i].weight*((i+1)==len(self.model.layers))

            # Update decays d
            self.si[i] = self.learning_rate * (self.si[i]) + (1 - self.learning_rate) * self.dwi ** 2

        self.model.weights = [layer.weight for layer in self.model.layers if layer.weight is not None]
		
	
		
		


class adam(optimizer):
    """
    this is initialization fn for adam
    argument: model, loss_fn, learning_rate, beta1,beta2,epsilon
    """

    def __init__(self, model,loss_fn, learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-8):

        self.learning_rate = learning_rate
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon=epsilon
        self.loss_fn = loss_fn
        self.v = []
        self.s = []
        self.v_corrected = []
        self.s_corrected = []
        L = len(self.model.layers)
        for i in range(L):

            self.v.append(np.zeros_like(self.model.layers[i].weight))  # set v =0 in all layers
            self.s.append(np.zeros_like(self.model.layers[i].weight))
            self.v_corrected.append(np.zeros_like(self.model.layers[i].weight))
            self.s_corrected.append(np.zeros_like(self.model.layers[i].weight))
			
			
			

	

    def step(self):
        """
        this function update weight according to adam
        """
        grad_loss = self.loss_fn.backward()
        last_dx = grad_loss.T

        for i in reversed(range(len(self.model.layers))):
            dw, dx = self.model.layers[i].backward(last_dx)
            last_dx = dx
            # Update Rule
            if self.model.layers[i].is_activation:
                continue
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            t = (self.model.history['epoch_number']+1)
            self.v[i] = self.beta1 * self.v[i] + (1 - self.beta1 + self.epsilon) * dw
            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            self.v_corrected[i] = self.v[i] / (1 - np.power(self.beta1, t) + self.epsilon)
            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            self.s[i] = self.beta2 * self.s[i] + (1 - self.beta2 + self.epsilon) * np.power(dw, 2)
            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            self.s_corrected[i] = self.s[i] / (1 - np.power(self.beta2 , t) + self.epsilon)
            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
            self.model.layers[i].weight = self.model.layers[i].weight - self.learning_rate * self.v_corrected[i] / np.sqrt(self.s[i] + self.epsilon) - self.loss_fn.lamda*self.model.layers[i].weight*((i+1)==len(self.model.layers))

        self.model.weights = [layer.weight for layer in self.model.layers if layer.weight is not None]
		

		
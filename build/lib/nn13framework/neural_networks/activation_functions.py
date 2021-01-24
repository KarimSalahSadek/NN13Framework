#ALL FUNCTIONS TAKE NUMPY ARRAYS AS INPUT AND RETURN NUMPY ARRAYS!!!!
#Numpy will automatically understand that 1x1 arrays are not arrays and return them correctly
#If grad is set to True , return the derivative instead
import numpy as np



def sigmoid(x , grad = False):
    '''

      :param z: a vector of the input x
      :param grad: a boolean to determine whether the output is the sigmoid or its derivative
      :return: if grad equals True the output is sum of e^(xi)/sum of(e^(allx))
              if grad equals False the output is the derivative = a tensor containing a matrix for each example where each matrix is every node derivative by all nodes
      '''
    # the function output
    if grad == False :
        return 1 / (1 + np.exp(-x))
    #  the function derivative
    elif grad== True:
        return sigmoid(x,False) * (1-sigmoid(x,False))



def relu(z,grad=False):
    '''

            :param z: a vector of the input x
            :param grad: a boolean to determine whether the output is the sigmoid or its derivative
            :return: if grad equals True the output is argmax(0,x)
                    if grad equals False the output is the derivative = 0 when x<=0 , 1 when x>0
    '''
    # the function output
    if grad == False:
        return np.where(z<=0,0,z)
    # the function derivative
    elif grad == True:
        z=np.where(z>0,1,z)
        return np.where(z <= 0, 0, z)


def softmax(z,grad=False):
    '''

       :param z: a vector of the input x
       :param grad: a boolean to determine whether the output is the sigmoid or its derivative
       :return: if grad equals True the output is sum of e^(xi)/sum of(e^(allx))
               if grad equals False the output is the derivative = a tensor containing a matrix for each example where each matrix is every node derivative by all nodes
    '''
    z=np.array(z)
    # the function output
    if grad == False:
        pos_condition = np.max(z,axis=0,keepdims=True) > 0
        z = z - np.max(z,axis=0,keepdims=True)*pos_condition
        z = np.exp(z)
        return z/(np.sum(z,axis=0,keepdims=True)+np.exp(-150))
    # the function derivative
    elif grad == True:
        z=z.T
        no_of_examples, no_of_classes = z.shape
        p = softmax(z)

        tensor1 = np.einsum('ij,ik->ijk', p, p)  # (m, n, n)

        tensor2 = np.einsum('ij,jk->ijk', p, np.eye(no_of_classes, no_of_classes))  # (m, n, n)


        return tensor2 - tensor1


def tanh(pred , grad = False):
    """
        tanh(x)= (e^(2*x) - 1)/(e^(2*x) + 1)
        grad = (1 - tanh^(2))
        parmetersare : 
        pred : Matrix represent the output of a linear layer
        gred : Flag to determine which to calculate the function or its drevative 
    """
    # Tanh 's output
    if grad == False :
        return (np.exp(2*pred) - 1) / (np.exp(2* pred) + 1 +np.exp(-150))
    # Tanh 's derivative 
    elif grad== True:
        return (1 - np.square(tanh(pred,False)))



def hard_tanh(pred, grad = False):
    """
        Hard tanh(x) = max(min(x,1),-1)  if (x > 1) {f(x)= 1} elif(x < -1) {f(x) = -1} else {f(x) = x}
        grad =   if (x > 1) {f'(x)= 0} elif(x < -1) {f'(x) = 0} else {f(x) = 1}
        parmetersare : 
        pred : Matrix represent the output of a linear layer
        gred : Flag to determine which to calculate the function or its drevative
    """
    #HardTanh output
    X = pred
    if grad == False:
        X = np.where(X >  1,  1,X) 
        X = np.where(X < -1, -1,X)
        return X
    #HardTanh grad 
    elif grad == True:
        X = np.where(X >=  1,  0,X) 
        X = np.where(X <= -1,  0,X)
        X= np.where(X!=0, 1 , 0)
        return X



def sign(pred , grad = False):
    """
        sign act fun
        grad = zeros with dimension like pred 
        parmetersare : 
        pred : Matrix represent the output of a linear layer
        gred : Flag to determine which to calculate the function or its drevative
    """
    X = np.zeros_like(pred)
    # sign 's output
    if grad == False :
        return (np.sign(pred))
    # sign 's derivative 
    elif grad== True:
        return X


def leaky_relu(pred, grad = False):
    """
        leaky relu activation function 
        if X > 0   L_relu(X) = X    otherwise     L_relu(X) = alpha * X     alpha = 0.01  mostly 
        if X > 0   gred = 1         otherwise     gred = alpha              alpha = 0.01  mostly      
        parmetersare : 
        pred : Matrix represent the output of a linear layer
        gred : Flag to determine which to calculate the function or its drevative
    """
    X = np.zeros_like(pred)
    alpha = 0.01 
    # leaky_relu 's out
    if grad == False:
        return np.where(pred <= 0,np.multiply(alpha,pred),pred)
    #leaky_relu 's drevative
    elif grad == True:
        X = np.where(pred <= 0,alpha,1)
        return X 



def elu(pred,grad=False):
    """
        elu activation function
        if X > 0   elu(x) = x      otherwise elu(x) = alpha * (exp(X) - 1)
        if X > 0   gred(x) = 1     otherwise gred = alpha * exp(X)
        parmetersare : 
        pred : Matrix represent the output of a linear layer
        gred : Flag to determine which to calculate the function or its drevative
    """
    X = np.zeros_like(pred) 
    alpha = 0.01 
    # elu 's out
    if grad == False:
        return np.where(pred <= 0, np.multiply(alpha, (np.exp(pred) - 1)), pred)
    # elu 's drevative
    elif grad == True:
        X = np.where(pred <= 0, np.multiply(alpha, np.exp(pred)) ,1)
        return X 




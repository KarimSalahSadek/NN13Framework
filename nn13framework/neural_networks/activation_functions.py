#ALL FUNCTIONS TAKE NUMPY ARRAYS AS INPUT AND RETURN NUMPY ARRAYS!!!!
#Numpy will automatically understand that 1x1 arrays are not arrays and return them correctly
#If grad is set to True , return the derivative instead
import numpy as np



def sigmoid(x , grad = False):
    # the function output
    if grad == False :
        return 1 / (1 + np.exp(-x))
    #  the function derivative
    elif grad== True:
        return sigmoid(x,False) * (1-sigmoid(x,False))



def relu(z,grad=False):
    # the function output
    if grad == False:
        return np.where(z<=0,0,z)
    # the function derivative
    elif grad == True:
        z=np.where(z>0,1,z)
        return np.where(z <= 0, 0, z)


def softmax(z,grad=False):
    z=np.array(z)
    # the function output
    if grad == False:
        z=z-np.abs(np.max(z,axis=0,keepdims=True))
        z= np.exp(z)
        return z/(np.sum(z,axis=0,keepdims=True)+10e-6)
    # the function derivative
    elif grad == True:
        z=z.T
        no_of_examples, no_of_classes = z.shape
        p = softmax(z)

        tensor1 = np.einsum('ij,ik->ijk', p, p)  # (m, n, n)

        tensor2 = np.einsum('ij,jk->ijk', p, np.eye(no_of_classes, no_of_classes))  # (m, n, n)


        return tensor2 - tensor1





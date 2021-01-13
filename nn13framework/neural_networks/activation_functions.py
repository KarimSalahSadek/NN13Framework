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
        z=z-np.max(z,axis=1,keepdims=True)
        z= np.exp(z)
        return z/np.sum(z,axis=1,keepdims=True)
    # the function derivative
    elif grad == True:
        # return the derivative for each class by d itself
        return softmax(z,False)*(1-softmax(z,False))






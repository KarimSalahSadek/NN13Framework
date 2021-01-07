#ALL FUNCTIONS TAKE NUMPY ARRAYS AS INPUT AND RETURN NUMPY ARRAYS!!!!
#Numpy will automatically understand that 1x1 arrays are not arrays and return them correctly
#All Funcitons are static methods that don't require an instance to call

#If grad is set to True , return the derivative instead
import numpy as np

# @staticmethod
def sigmoid(x , grad = False):
    # the function output
    if grad == False :
        return 1 / (1 + np.exp(-x))
    #  the function derivative
    elif grad== True:
        return sigmoid(x,False) * (1-sigmoid(x,False))


# @staticmethod
def relu(z,grad=False):
    # the function output
    if grad == False:
        return np.where(z<=0,0,z)
    # the function derivative
    elif grad == True:
        z=np.where(z>0,1,z)
        return np.where(z <= 0, 0, z)


# @staticmethod
def softmax(z,grad=False):
    # the function output
    if grad == False:
        return np.where(z<=0,0,z)
    # the function derivative
    elif grad == True:
        z=np.where(z>0,1,z)
        return np.where(z <= 0, 0, z)







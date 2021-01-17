#ALL FUNCTIONS TAKE NUMPY ARRAYS AS INPUT AND RETURN NUMPY ARRAYS!!!!
#Numpy will automatically understand that 1x1 arrays are not arrays and return them correctly
#If grad is set to True , return the derivative instead
import numpy as np
from scipy.ndimage.interpolation import shift


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
    z2=np.array(z)
    # the function output
    if grad == False:
        z=z-np.max(z,axis=1,keepdims=True)
        z= np.exp(z)
        return z/np.sum(z,axis=1,keepdims=True)
    # the function derivative
    elif grad == True:
        # get number of examples = row_len , examle dimension = col_len
        col_len, row_len = z.shape
        # create a 3d matrix
        grad_of_all_examples =np.zeros((row_len,col_len,col_len))
        unity_matrix =np.identity(col_len)
        for i in range(row_len):
            #create x1 , x2 to hold the ith example
            x1=np.zeros((col_len,col_len))
            x2= np.zeros((col_len, col_len))
            x3= np.zeros((col_len, col_len))
            for j in range(col_len):
                x1[:,[j]]=z[:, [i]]
                #shift each x2 column
                x2[:,[j]]=np.roll(z2[:, [i]], -j, axis = 0)
            x3=np.array(x2)
            # shift each x2 row in x3
            for j in range(col_len):
                for k in range(col_len):
                    x3[j,k]=x2[j,k-j]
            grad_of_all_examples[i]=(softmax(x1.T, False) * (unity_matrix - softmax(x3.T, False))).T
        #return the final tensor
        return np.sum(grad_of_all_examples,axis=0)
        # return (softmax(z.T,False)*(1-softmax(z.T,False))).T






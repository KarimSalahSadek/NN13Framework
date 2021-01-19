import pickle
# import random
import numpy as np
import os
import tarfile
from urllib import request
import gzip
from urllib.request import urlretrieve
import matplotlib.pyplot  as plt
import math


############################################# FOR GENERIC DATA ###############################################################


def one_hot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    
    Z[np.arange(N), Y] = 1 #Z[][]==Z[,]
    return Z


def get_batch_generator_XY(X,Y,batch_size,shuffle=False):
    if shuffle:
        X,Y=shuffle_data(X,Y)
    N=len(Y)
    batch_number=math.ceil(N/batch_size)
    count=0
    for i in range (0,N,batch_size):
        #to prevent last batch to be less than the batch size
        if (N%batch_size)!=0 and count==batch_number-1:
            break
        else :
            yield X[i:i+batch_size],Y[i:i+batch_size]
            count+=1
   

def get_batch_XY(X_Data,Y_Data,batch_size,shuffle=False):
    X_batch_generator,Y_batch_generator=zip(*get_batch_generator_XY(X_Data,Y_Data,batch_size,shuffle))
    X_batches=list()
    Y_batches=list()
    for x in X_batch_generator:
        X_batches.append(x)
    for y in Y_batch_generator:
        Y_batches.append(y)
    
    return X_batches,Y_batches




#MinMaxScaler Normaliztion
def init_norm(data):
    min_data=np.min(data)
    max_data=np.max(data)
    output_data=(data-min_data)/(max_data-min_data)
    return output_data


#Data Standardization (StandardScaler)
def init_standard(data):
    mean=np.mean(data)
    standard_div=np.std(data)
    output_data=(data-mean)/standard_div
    return output_data

#Shuffle the data if needed
def shuffle_data(X,Y):
    #By default X.shape[0] is the size of the numpy array given or passed to the function
    perm=np.random.permutation(len(Y))
    X_shuffled=X[perm]
    Y_shuffled=Y[perm]
    return X_shuffled,Y_shuffled


def train_test_split(X,Y,test_size):
    #get the size of the dataset passed to the function
    data_size=X.shape[0]
    if test_size >1 :
        return "ERROR , the test size can not be larger than 1"
    else :
    #calculate the test size and train size
        data_test_size=int(data_size*test_size)
        data_train_size=data_size-data_test_size
        #Shuffle the data 
        X_shuffled,Y_shuffled=shuffle_data(X,Y)
        #Train_test for X,Y
        X_train=X_shuffled[0:data_train_size-1]
        X_test=X_shuffled[data_train_size:data_train_size+data_test_size]
        Y_train=Y_shuffled[0:data_train_size-1]
        Y_test=Y_shuffled[data_train_size:data_train_size+data_test_size]
        return X_train,Y_train,X_test,Y_test




############################################# FOR MNIST DATA ###############################################################


def download_save_mnist(path=None,CNN=False):

    if CNN==None :
        CNN=False

    Mnist_files = [
	["training_images","train-images-idx3-ubyte.gz",9912422],
	["test_images","t10k-images-idx3-ubyte.gz",1648877],
	["training_labels","train-labels-idx1-ubyte.gz",28881],
	["test_labels","t10k-labels-idx1-ubyte.gz",4542]
    ]

    mnist_url = "http://yann.lecun.com/exdb/mnist/"

    # Set path to /home/USER_NAME/data/Mnist or C:\Users\USER_NAME\data\Mnist
    if path is None:
        path = os.path.join(os.path.expanduser('~'), 'data', 'Mnist')
    
    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    count=0
    #Download files
    for name in Mnist_files:
        if name[1] not in os.listdir(path):
            print("Downloading "+name[1]+"...")
            count+=1
            request.urlretrieve(mnist_url+name[1],os.path.join(path, name[1]))

    #if the download is not completed so that the downloaded file be deleted and then re-downloaded
        elif name[1] in os.listdir(path) and os.path.getsize(os.path.join(path, name[1]))!=name[2]:
            os.remove(os.path.join(path, name[1]))
            print("Downloading "+name[1]+"...")
            count+=1
            request.urlretrieve(mnist_url+name[1],os.path.join(path, name[1]))

    if count==0:
        print("Mnist dataset is already downloaded in the directory :",path)
    else:
        print("Download complete.")
########################################################################################## SAVING #############################################################################################
    #saving the dataset
    mnist = {}
    
    #save inputs
    for name in Mnist_files[:2]:
        with gzip.open(os.path.join(path,name[1]), 'rb') as f:
            if CNN:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28,28,1)
            else:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    
    #save labels
    for name in Mnist_files[-2:]:
        with gzip.open(os.path.join(path,name[1]), 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    with open(path+"/mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete in the directory :", path)





def load_mnist(path=None):

    # Set path to /home/USER/data/mnist or C:\Users\USER_NAME\data\mnist
    if path is None:
        path = os.path.join(os.path.expanduser('~'), 'data', 'Mnist')
    #Open Pickle file
    with open(path+"/mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def load_mnist_data(path=None,CNN=False):
    if CNN==None:
        CNN=False
    #downloading & and saving
    download_save_mnist(path,CNN)
    #loading
    X_train,Y_train,X_test,Y_test=load_mnist(path)
    #Nomralization & One hot encoding
    X_train, X_test = X_train/float(255), X_test/float(255)
    Data_out=10
    Y_train=one_hot(Y_train, Data_out)
    Y_test=one_hot(Y_test,Data_out)

    return X_train,Y_train,X_test,Y_test











##################################################################### FOR TESTING ########################################################################

# load_mnist_data(path=None,CNN=True)

 
#Steps for loading Mnist step by step (Not required to do that now)
##  1st : download & save the dataset 
# download_save_mnist()
## 2nd : Load it in dictionaries of names training_images,test_images,training_labels,and test_labels
# X_train, Y_train, X_test, Y_test=load_mnist()
##3rd : Normaliztion
# X_train, X_test = X_train/float(255), X_test/float(255)
##5th : convert Y into one hot encoding
## DataOut= is the number of classes to be applied in the output in order to make proper one hot encoding  
##for testing the mnist the Data_out will be 10
# Data_out=10
# Y_train=one_hot(Y_train, Data_out)
# Y_test=one_hot(Y_test,Data_out)

###################################################### Steps for batch & CNN DATA LOADER ############################################

#Please use it like this way 


#NOTE : X_batches_list is not required at all for your testing it is only for me to test if the functions is functional 

# X_train,Y_train,X_test,Y_test=load_mnist_data(path=None,CNN=False)


# #TESTING THE CNN DATA LOADER & Compare the output in first examples
# print(Y_train[8])
# #plt.figure()
# plt.imshow(X_train[8])
# plt.show()  # display it


# #MORE EXAMPLES , CLOSE THE WINDOW OF THE FIRST EXAMPLE FIRST
# print(Y_train[10])
# #plt.figure()
# plt.imshow(X_train[10])
# plt.show()  # display it

################################################ 1st test ####################################################

# X_batch_test=[]
# batch_size=4957
# for epoch in range (0,50):
#         X_batches,y_batches=get_batch_XY(X_train,Y_train,batch_size,shuffle=False)
#         #to compare between different batches in an epoch
#         if epoch==1 : 
#             X_batch_test.append(X_batches[1])
#             X_batch_test.append(X_batches[5])
#         #print(len(X_batches))  #will be 12 in this example
#         #print(X_batches[0].shape)



################################################ 2nd test ####################################################
# X_batch_test=[]
# batch_size=4957
# X_batches,y_batches=get_batch_XY(X_train,Y_train,batch_size,shuffle=False)
# for epoch in range (0,50):
#          #to compare between different batches in an epoch
#          if epoch==1 : 
#              X_batch_test.append(X_batches[1])
#              X_batch_test.append(X_batches[5])
#          print(len(X_batches))  #will be 12 in this example
#          print(X_batches[0].shape)


#1st method to keep tracking , is 5anzara method by keep tracking by your eye to the first changed element in lets say 3rd col in the second example in batch 1 and compare it to the second ex in batch2 
# print(X_batch_test[0])
# print("***********************************************************************************************************************")
# print(X_batch_test[1])

#2nd method is to be a human and use a numpy function to compare between 2 numpy arrays (There is more than one numpy array)
# print(np.array_equal(X_batch_test[0],X_batch_test[1]))

#https://www.geeksforgeeks.org/how-to-compare-two-numpy-arrays/ This method only tnf3 lma yb2a 3ndk array of size so8ir msh 64,784

########################## STEPS FOR CNN DATA_LOADER ####################################################
# BY DEFAULT THE DATA_LOADER WILL RETURN ONLY THE DATA IN THIS SHAPE (60000,784) IF IT IS TRAINING DATA , BUT IF YOU WANT TO USE CNN , JUST CALL THE FUNCTION load_mnist_data 
# AND PUT CNN=TRUE or IF YOU WANT TO GO FOR STEP BY STEP , JUST USE THIS download_save_mnist AND PUT CNN=TRUE


# THE DATALOADER IS ATGRABET BY ME USING MATPLOTLIB IN MORE THAN 10 EXAMPLES AND IT WORKS WELL
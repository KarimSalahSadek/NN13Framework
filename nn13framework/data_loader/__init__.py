import pickle
import random
import numpy as np
import os
import tarfile
from urllib import request
import gzip
from urllib.request import urlretrieve





############################################# FOR GENERIC DATA ###############################################################


def one_hot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z


def get_batch_generator_X(X,batch_size):
    i=0
    N=len(X)
    if i*batch_size>=N:
        i=0
    else:
        X = X[i*batch_size:(i+1)*batch_size,:]
        i += 1
    yield X 

def get_batch_generator_Y(Y,batch_size):
    i=0
    N=len(Y)
    if i*batch_size>=N:
        i=0
    else:
        Y = Y[i*batch_size:(i+1)*batch_size]
        i += 1
    yield Y

def get_batch_XY(X_batch_generator,Y_batch_generator):
    for x in X_batch_generator:
        X_batch=x
    for y in Y_batch_generator:
        Y_batch=y
    return X_batch,Y_batch



#MinMaxScaler
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
        print(X_shuffled.shape)
        print(Y_shuffled.shape)
        #Train_test for X,Y
        X_train=X_shuffled[0:data_train_size-1]
        X_test=X_shuffled[data_train_size:data_train_size+data_test_size]
        Y_train=Y_shuffled[0:data_train_size-1]
        Y_test=Y_shuffled[data_train_size:data_train_size+data_test_size]

        return X_train,Y_train,X_test,Y_test



############################################# FOR CIFAR10 DATA ###############################################################







############################################# FOR MNIST DATA ###############################################################




# path=os.path.join(os.path.expanduser('~'), 'data', 'Mnist')
Mnist_files = [
	["training_images","train-images-idx3-ubyte.gz",9912422],
	["test_images","t10k-images-idx3-ubyte.gz",1648877],
	["training_labels","train-labels-idx1-ubyte.gz",28881],
	["test_labels","t10k-labels-idx1-ubyte.gz",4542]
]



def download_mnist(path=None):

    mnist_url = "http://yann.lecun.com/exdb/mnist/"
    if path is None:
# Set path to /home/USER/data/mnist or C:\Users\USER_NAME\data\mnist
        path = os.path.join(os.path.expanduser('~'), 'data', 'Mnist')
    
# Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)
    count=0
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

#Loading the dataset
    mnist = {}
    for name in Mnist_files[:2]:
        with gzip.open(os.path.join(path,name[1]), 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in Mnist_files[-2:]:
        with gzip.open(os.path.join(path,name[1]), 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open(path+"/mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete in the directory :", path)






def load_mnist(path=None):
    with open(path+"/mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]




def load_mnist_data(path=None):
    download_mnist(path)
    X_train,X_test,Y_train,Y_test=load_mnist(path)





path='C:/Users/Kiro/Desktop'
download_mnist(path)
x_train,y_train,x_test,y_test=load_mnist(path)
y_train=one_hot(y_train,10)
train_test_split(x_train,y_train,test_size=0.333)


#Steps 
##  1st : download the dataset 
# download_mnist()
## 2nd : save it in the same directory as pickle file
# save_mnist()
## 3rd : Load it in dictionaries of names training_images,test_images,training_labels,and test_labels
# X_train, Y_train, X_test, Y_test=load()
##4th : Normaliztion
# X_train, X_test = X_train/float(255), X_test/float(255)
##5th : convert the X dictionaries into numpy arrays , and Y into one hot encoding
## DataOut= is the number of classes to be applied in the output in order to make proper one hot encoding  
##for testing the mnist the Data_out will be 10
# Data_out=10
# X_train -= np.mean(X_train)
# X_test -= np.mean(X_test)
# Y_train=one_hot(Y_train, Data_out)
# Y_test=one_hot(Y_test,Data_out)
##6th : in order to take batches from X and Y , so you need first to determine the size of the batch you need , then you will be returned a generator of X_batch and Y_batch
# X_batch_generator=get_batch_generator_X(X_train,batch_size)
# Y_batch_generator=get_batch_generator_Y(Y_train,batch_size)
##7th : convert the generator to X_batch & Y_batch
# X_batch,Y_batch=get_batch_XY(X_batch_generator,Y_batch_generator)

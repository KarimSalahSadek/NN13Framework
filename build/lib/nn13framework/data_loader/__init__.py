import pickle
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

    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing Mnist. Default is
            /home/USER/data/Mnist or C:\Users\USER\data\Mnist.
            Create if nonexistant. Download Mnist if missing.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values,
            with the order (red -> blue -> green). Columns of labels are a
            onehot encoding of the correct class.
    """


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
################################### SAVING ########################################
    #saving the dataset
    mnist = {}
    
    #save inputs
    for name in Mnist_files[:2]:
        with gzip.open(os.path.join(path,name[1]), 'rb') as f:
            if CNN:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,1,28,28)
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


def visualize_mnist(Data_input,idx):
    if idx>len(Data_input)-1:
       print("Out of index")
       return 0
    #reshaping back the dataset if the dataset shape is (len(data),1,28,28)
    Data_input=Data_input.reshape(len(Data_input),1,28,28).transpose(0,2,3,1)
    plt.figure()
    plt.imshow(Data_input[idx])
    plt.show()  # display it

            
############################################# FOR CIFAR DATA ###############################################################
def download_save_cifar10(path=None,CNN=False):
    if CNN==None :
        CNN=False    
    
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing CIFAR-10. Default is
            /home/USER/data/cifar10 or C:\Users\USER\data\cifar10.
            Create if nonexistant. Download CIFAR-10 if missing.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values,
            with the order (red -> blue -> green). Columns of labels are a
            onehot encoding of the correct class.
    """
    cifar_url = 'https://www.cs.toronto.edu/~kriz/'
    cifar_tar = 'cifar-10-binary.tar.gz'
    cifar_files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']
    
    Cifar_Size=170052171

    if path is None:
        # Set path to /home/USER/data/mnist or C:\Users\USER\data\mnist
        path = os.path.join(os.path.expanduser('~'), 'data', 'cifar10')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # # Download tarfile if missing
    # if cifar_tar not in os.listdir(path):
    #     urlretrieve(''.join((cifar_url, cifar_tar)), os.path.join(path, cifar_tar))
    #     print("Downloaded %s to %s" % (cifar_tar, path))

    
    
    # Download tarfile if missing
    if cifar_tar not in os.listdir(path): 
        print("Downloading cifar-10-binary.tar.gz....")
        urlretrieve(''.join((cifar_url, cifar_tar)), os.path.join(path, cifar_tar))
        print("Downloaded %s to %s" % (cifar_tar, path))
    else :
        #if the download is not completed so that the downloaded file be deleted and then re-downloaded
        if os.path.getsize(os.path.join(path, cifar_tar))!= Cifar_Size:
            os.remove(os.path.join(path,cifar_tar))
            print("Downloading cifar-10-binary.tar.gz....")
            urlretrieve(''.join((cifar_url, cifar_tar)), os.path.join(path, cifar_tar))
            print("Downloaded %s to %s" % (cifar_tar, path))
        else :
            print("Cifar10 dataset is already downloaded in the directory :",path)

    # Load data from tarfile
    with tarfile.open(os.path.join(path, cifar_tar)) as tar_object:
        # Each file contains 10,000 color images and 10,000 labels
        fsize = 10000 * (32 * 32 * 3) + 10000

        # There are 6 files (5 train and 1 test)
        buffr = np.zeros(fsize * 6, dtype='uint8')

        # Get members of tar corresponding to data files
        # -- The tar contains README's and other extraneous stuff
        members = [file for file in tar_object if file.name in cifar_files]

        # Sort those members by name
        # -- Ensures we load train data in the proper order
        # -- Ensures that test data is the last file in the list
        members.sort(key=lambda member: member.name)

        # Extract data from members
        for i, member in enumerate(members):
            # Get member as a file object
            f = tar_object.extractfile(member)
            # Read bytes from that file object into buffr
            buffr[i * fsize:(i + 1) * fsize] = np.frombuffer(f.read(), 'B')

    # Parse data from buffer
    # -- Examples are in chunks of 3,073 bytes
    # -- First byte of each chunk is the label
    # -- Next 32 * 32 * 3 = 3,072 bytes are its corresponding image

    # Labels are the first byte of every chunk
    labels = buffr[::3073]

    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
    images = pixels.reshape(-1, 3072).astype('float32') / 255

    # Split into train and test
    train_images, test_images = images[:50000], images[50000:]
    train_labels, test_labels = labels[:50000], labels[50000:]

    if CNN==True:
        train_images=train_images.reshape(len(train_images),3,32,32)
        test_images=train_images.reshape(len(train_images),3,32,32)


    return train_images, one_hot(train_labels,10),test_images, one_hot(test_labels,10)



def visualize_cifar(Data_input,idx):
    if idx>len(Data_input)-1:
       print("Out of index")
       return 0
    #reshaping back the dataset if the dataset shape is (len(data),1,28,28)
    Data_input=Data_input.reshape(len(Data_input),3,32,32).transpose(0,2,3,1)
    plt.figure()
    plt.imshow(Data_input[idx])
    plt.show()  # display it

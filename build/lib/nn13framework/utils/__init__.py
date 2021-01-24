import pickle as pkl
import numpy as np

def save_model(filename,model):
    """
        Creates a file with the given name and saves the model in it
        Training history is not included in the file
        Warning: Resulting file will be large (100s of MBs)
    """
    hist = model.history
    model_copy = model
    model_copy.reset_history()
    f = open(filename,'wb')
    pkl.dump(model_copy,f)
    f.close()
    model.history = hist

def load_model(filename):
    """
        Loads a model from the given file
    """
    model = None
    f = open(filename,'rb')
    model = pkl.load(f)
    f.close()
    return model

def save_weights(filename,model):
    """
        Creates a file with the given name and saves the model parameters (model.weights) in it
    """
    f = open(filename,'wb')
    np.save(f,np.array(model.weights))
    f.close()

def load_weights(filename):
    """
        Loads parameters list from the given file
    """    
    f = open(filename,'rb')
    weights = np.load(f,allow_pickle=True)
    f.close()
    return weights.tolist()
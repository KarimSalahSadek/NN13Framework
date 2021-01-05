#All Funcitons are static methods that don't require an instance to call them

#Assume TP,TN,FP,FN are all included in the model.history dict
#In every function, Check first if model is an instance of the neural_network class, call assert on result
@staticmethod
def recall(model):
    true_positives = model.history['true_positives']
    pass

    return None
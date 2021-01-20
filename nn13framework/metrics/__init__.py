#Assume TP,TN,FP,FN are all included in the model.history dict
#In every function, Check first if model is an instance of the neural_network class, call assert on result

def recall(model):
    tp = model.history['true_positives']
    tn = model.history['true_negatives']
    fp = model.history['false_positives']
    fn = model.history['false_negatives']
    recall = tp/(tp+fn)
    return recall

def precision(model):
    tp = model.history['true_positives']
    tn = model.history['true_negatives']
    fp = model.history['false_positives']
    fn = model.history['false_negatives']
    precision = tp/(tp+fp)
    return precision

def accuracy(model):
    tp = model.history['true_positives']
    tn = model.history['true_negatives']
    fp = model.history['false_positives']
    fn = model.history['false_negatives']
    accuracy = (tp+tn)/(tp+fn+tn+fp)
    return accuracy

def f1score(model):
    precision_ = precision(model)
    recall_ = recall(model)
    f1score = 2*precision_*recall_/(precision_+recall_)
    return f1score
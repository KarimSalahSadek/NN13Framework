#Assume TP,TN,FP,FN are all included in the model.history dict
#In every function, Check first if model is an instance of the neural_network class, call assert on result

def recall(model,class_index):
    tp = model.history['true_positives'][class_index]
    tn = model.history['true_negatives'][class_index]
    fp = model.history['false_positives'][class_index]
    fn = model.history['false_negatives'][class_index]
    recall = tp/(tp+fn)
    return recall

def precision(model,class_index):
    tp = model.history['true_positives'][class_index]
    tn = model.history['true_negatives'][class_index]
    fp = model.history['false_positives'][class_index]
    fn = model.history['false_negatives'][class_index]
    precision = tp/(tp+fp)
    return precision

def class_accuracy(model,class_index):
    tp = model.history['true_positives'][class_index]
    tn = model.history['true_negatives'][class_index]
    fp = model.history['false_positives'][class_index]
    fn = model.history['false_negatives'][class_index]
    accuracy = (tp+tn)/(tp+fn+tn+fp)
    return accuracy

def f1score(model,class_index):
    precision_ = precision(model,class_index)
    recall_ = recall(model,class_index)
    f1score = 2*precision_*recall_/(precision_+recall_)
    return f1score
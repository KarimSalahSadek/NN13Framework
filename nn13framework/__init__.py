
#Tests a condition and returns true to stop training if condtion is met
def stopping_function(metric,value):
    if metric is None:
        return False
    if value is None:
        return False

    pass

    return None

#Main training function
def train(model,data,number_of_epochs,optimizer,learning_rate,stopping_function_metric=None,stopping_function_value=None):
    pass

#Returns a dictionary of metrics
def test(model,test_data,metric = None):
    metrics = {}
    metrics['accuracy'] = 0
    metrics['recall'] = 0
    pass

    if metric == None:
        return metrics
    else:
        return metrics['metric']

#Uses model to solve one piece of input
def use_model(model,data_piece):
    predicted_output = None

    return predicted_output
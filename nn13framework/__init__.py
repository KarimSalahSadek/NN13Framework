import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

#Tests a condition and returns true to stop training if condtion is met
def stopping_function(metric,value):
    if metric is None:
        return False
    if value is None:
        return False

    pass

    return None

#Single batch training
def iteration(index,model,data,data_val,criterion,optimizer,print_every,vis,Visualization_List=None):
    print("Epoch ",index+1,':')
    epoch_loss = 0
    accuracy = 0
    step = 0
    model.set_evaluate_mode(False)
    for X,Y in zip(data[0],data[1]):
        step +=1
        out = model.forward(X)
        loss = criterion.evaluate(out,Y)/out.shape[0]
        optimizer.step()
        epoch_loss = loss
        if step%print_every == 0:
            print(' Training Loss = ',round(loss,5))
        model.history['batch_loss'].append(loss)
        pred = np.argmax(out,1)
        label = np.argmax(Y,1)
        accuracy += np.sum(pred==label)/out.shape[0]
        num_labels = Y.shape[1]
        pred_label_matrix = np.zeros((num_labels,num_labels))
        for i in range(Y.shape[0]):
            pred_label_matrix[label[i]][pred[i]]+=1
        for i in range(num_labels):
            model.history['true_positives'][i] += pred_label_matrix[i][i]
            temp = pred_label_matrix
            temp[i] -= temp[i]
            temp[:,i] -= temp[:,i]
            model.history['true_negatives'][i] += np.sum(temp)
            model.history['false_positives'][i] += np.sum(pred_label_matrix[:,i] - pred_label_matrix[i,i])
            model.history['false_negatives'][i] += np.sum(pred_label_matrix[i] - pred_label_matrix[i,i])
    num_batches = len(data[1])
    accuracy = accuracy/num_batches
    model.history['epoch_loss'].append(epoch_loss)
    model.history['accuracy'].append(accuracy)
    X_val , Y_val = next(zip(data_val[0],data_val[1]))
    model.set_evaluate_mode(True)
    val_out = model.forward(X_val)
    val_loss = criterion.evaluate(val_out,Y_val)/val_out.shape[0]
    pred = np.argmax(val_out,1)
    label = np.argmax(Y_val,1)
    val_acc = np.sum(pred==label)/val_out.shape[0]
    model.history['validation_loss'].append(val_loss)
    model.history['validation_accuracy'].append(val_acc)
    print("Epoch end:")
    print(" Last Loss = " + str(round(epoch_loss,5)) + "\t\tValidation Loss = " + str(round(val_loss,5)))
    print(" Training Accuracy = " + str(round(accuracy,2)) + "\t\tValidation Accuracy = " + str(round(val_acc,2)))
    if vis =='animated':
        Visualization_List[1].append(index)
        Visualization_List[2].set_data(Visualization_List[1],model.history['epoch_loss'])
        Visualization_List[3].set_data(Visualization_List[1],model.history['validation_loss'])
        Visualization_List[4].set_data(Visualization_List[1],model.history['accuracy'])
        Visualization_List[5].set_data(Visualization_List[1],model.history['validation_accuracy'])
        Visualization_List[0][0].relim()
        Visualization_List[0][1].relim()
        Visualization_List[0][0].autoscale_view()
        Visualization_List[0][1].autoscale_view()

def Init():
    pass

#Main training function
def train(model,data,validation_data,epochs,criterion,optimizer,reset_history=False,visualization='static',stopping_function_metric=None,stopping_function_value=None,loss_prints_per_epoch = 5):
    if len(model.history['true_positives']) == 0:
        model.history['true_positives'] = [0]*data[1][0].shape[1]
    if len(model.history['false_positives']) == 0:
        model.history['false_positives'] = [0]*data[1][0].shape[1]
    if len(model.history['true_negatives']) == 0:
        model.history['true_negatives'] = [0]*data[1][0].shape[1]
    if len(model.history['false_negatives']) == 0:
        model.history['false_negatives'] = [0]*data[1][0].shape[1]
    print_every = len(data[1])//loss_prints_per_epoch
    if(visualization == 'animated'):
        fig, axs = plt.subplots(2, 1,figsize=(10,10))
        x_ax = []
        ln1, = axs[0].plot([], [], 'r', label='Loss')
        ln2, = axs[0].plot([], [], 'm', label='Valid. Loss')
        ln3, = axs[1].plot([], [], 'g', label='Accuracy')
        ln4, = axs[1].plot([], [], 'b', label='Valid. Accuracy')
        axs[0].legend(loc='upper right', frameon=False)
        axs[1].legend(loc='lower right', frameon=False)
        axs[0].set_xlabel('Loss',fontweight='bold')
        axs[1].set_xlabel('Accuracy',fontweight='bold')
        axs[0].set_title('Live Graphs',fontweight='bold')
        axs[0].set_xlim([0,epochs-1])
        axs[1].set_xlim([0,epochs-1])
        axs[1].set_ylim([0,1])
        func = iteration
        visualization_args=[axs,x_ax,ln1,ln2,ln3,ln4]
        args = (model,data,validation_data,criterion,optimizer,print_every,visualization,visualization_args)
        Pic = matplotlib.animation.FuncAnimation(fig, func,frames=epochs,repeat=False,fargs=args,init_func=Init)
        plt.show()
        writer = matplotlib.animation.PillowWriter(fps=1)
        Pic.save("Live_GIF.gif", writer=writer)
        print('Training Finished!')
    elif(visualization == 'static'):
        for epoch in range(epochs):
            iteration(epoch,model,data,validation_data,criterion,optimizer,print_every,visualization)
        print('Training Finished!')
        #STATIC VIS FUNCS
    elif(visualization == 'text' or visualization is None):
        for epoch in range(epochs):
            iteration(epoch,model,data,validation_data,criterion,optimizer,print_every,visualization)
        print('Training Finished!')
    else:
        raise Exception('Wrong input for visulaization parameter in train function!')

#Returns a dictionary of metrics
def test(model,test_data,metric = None):
    metrics = {}
    metrics['accuracy'] = 0
    pass

    if metric == None:
        return metrics
    else:
        return metrics[metric]

#Uses model to solve one piece of input
def use_model(model,data_piece):
    predicted_output = None

    return predicted_output
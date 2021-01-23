# matplotlib functions that plots different types of graphs
# In every function, Check first if model is an instance of the neural_network class, call assert on result

# Assume that whatever parameters you want will be added to model.history dict later
# If disply_validation is True , then display both the metric for the validation set
# and the training set on the same graph
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import metrics as MT

Figure_Number = 0


def plot_accuracy_loss_vs_iterations(model, display_validation=True):
    global Figure_Number
    BatchLoss_fig = plt.figure(Figure_Number + 1)
    Figure_Number += 1
    plt.plot(model.history['batch_loss'], 'g', label='Batch Loss')
    plt.legend(loc='upper right', frameon=False)
    if display_validation:
        EpochLoss_fig = plt.figure(Figure_Number + 1)
        Figure_Number += 1
        plt.plot(model.history['epoch_loss'], 'b', label='Epoch Loss')
        plt.plot(model.history['validation_loss'], 'r', label='Validation Loss')
        plt.legend(loc='upper right', frameon=False)
        EpochAccuracy_fig = plt.figure(Figure_Number + 1)
        Figure_Number += 1
        plt.plot(model.history['accuracy'], 'm', label='Accuracy')
        plt.plot(model.history['validation_accuracy'], 'y', label='Validation Accuracy')
        plt.legend(loc='lower right', frameon=False)
    else:
        EpochLoss_fig = plt.figure(Figure_Number + 1)
        Figure_Number += 1
        plt.plot(model.history['epoch_loss'], 'b', label='Epoch Loss')
        plt.legend(loc='upper right', frameon=False)
        EpochAccuracy_fig = plt.figure(Figure_Number + 1)
        Figure_Number += 1
        plt.plot(model.history['accuracy'], 'm', label='Accuracy')
        plt.legend(loc='lower right', frameon=False)


def plot_img(img_arr):
    global Figure_Number
    img_fig = plt.figure(Figure_Number + 1)
    Figure_Number += 1
    img_2d = img_arr.reshape(28, 28)
    plt.imshow(img_2d)


def plot_confusion_matrix(Class_number, model):
    global Figure_Number
    Confusion_fig = plt.figure(Figure_Number + 1)
    Figure_Number += 1
    Percentages = []
    Labels = []
    if model.history['true_positives'][Class_number] > 0:
        Percentages.append(model.history['true_positives'][Class_number])
        Labels.append('TP')
    if model.history['true_negatives'][Class_number] > 0:
        Percentages.append(model.history['true_negatives'][Class_number])
        Labels.append('TN')
    if model.history['false_positives'][Class_number] > 0:
        Percentages.append(model.history['false_positives'][Class_number])
        Labels.append('FP')
    if model.history['false_negatives'][Class_number] > 0:
        Percentages.append(model.history['false_negatives'][Class_number])
        Labels.append('FN')

    plt.pie(Percentages, autopct='%1.1f%%', labels=Labels, startangle=90)


def plot_metrics(Class_number, model):
    global Figure_Number
    recall=MT.recall(model,Class_number)
    precision=MT.precision(model,Class_number)
    Accuracy=MT.class_accuracy(model,Class_number)
    F1score=MT.f1score(model,Class_number)
    Metrics_fig=plt.figure(Figure_Number+1)
    Figure_Number=Figure_Number+1
    Labels=['Recall','Precision','Accuracy','F1Score']
    Values=[recall,precision,Accuracy,F1score]
    plt.bar(Labels,Values,width=0.5)
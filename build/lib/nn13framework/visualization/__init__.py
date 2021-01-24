import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import nn13framework.metrics as MT

Figure_Number = 0


def plot_accuracy_loss_vs_iterations(model, display_validation=True):
    '''

    :param model: it represents the model class which contain the history dict to plot loss & accuracy from .
    :param display_validation: it specifies if user want to plot the validation losss & accuracy or not
    :return: nothing, just plotting the loss & Accuracy in new Graphs.
    '''
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
    plt.show()

def plot_img(img_arr):
    '''

    :param img_arr: it represents the Image that user want to figure out it
    :return: nothing, just figure out the image.
    '''
    global Figure_Number
    img_fig = plt.figure(Figure_Number + 1)
    Figure_Number += 1
    img_2d = img_arr.reshape(28, 28)
    plt.imshow(img_2d)
    plt.show()


def plot_confusion_matrix(Class_number, model):
    '''

    :param Class_number: it represents the index of the class that user want to plot its Confusion Matrix.
    :param model: it represents the model class which contain the history dict to plot the Confusion Matrix From from.
    :return: nothing, just plotting the Confusion Matrix in a new Graph.
    '''
    global Figure_Number
    Confusion_fig = plt.figure(Figure_Number + 1)
    Figure_Number += 1
    Percentages = []
    Labels = []
    if model.history['true_positives'][Class_number] > 0:
        Percentages.append(model.history['true_positives'][Class_number])
    if model.history['true_negatives'][Class_number] > 0:
        Percentages.append(model.history['true_negatives'][Class_number])
    if model.history['false_positives'][Class_number] > 0:
        Percentages.append(model.history['false_positives'][Class_number])
    if model.history['false_negatives'][Class_number] > 0:
        Percentages.append(model.history['false_negatives'][Class_number])
    if model.history['true_positives'][Class_number] > 0:
        Labels.append('TP ({:.1f}%)'.format(100*(model.history['true_positives'][Class_number]/sum(Percentages))))
    if model.history['true_negatives'][Class_number] > 0:
        Labels.append('TN ({:.1f}%)'.format(100*(model.history['true_negatives'][Class_number]/sum(Percentages))))
    if model.history['false_positives'][Class_number] > 0:
        Labels.append('FP ({:.1f}%)'.format(100*(model.history['false_positives'][Class_number]/sum(Percentages))))
    if model.history['false_negatives'][Class_number] > 0:
        Labels.append('FN ({:.1f}%)'.format(100*(model.history['false_negatives'][Class_number]/sum(Percentages))))
    plt.pie(Percentages,startangle=90)
    plt.legend(Labels, loc=3)
    plt.show()


def plot_metrics(Class_number, model):
    '''

    :param Class_number: it represents the index of the class that user want to plot its Metrics params.
    :param model: it represents the model class which contain the history dict to plot the Metrics params from.
    :return: nothing, just plotting the Metrics Parameters in a new Graph.
    '''
    global Figure_Number
    recall=MT.recall(model,Class_number)
    precision=MT.precision(model,Class_number)
    Accuracy=MT.class_accuracy(model,Class_number)
    F1score=MT.f1score(model,Class_number)
    Metrics_fig=plt.figure(Figure_Number+1)
    Figure_Number=Figure_Number+1
    Labels=['Recall','Precision','Accuracy','F1Score']
    Labels_Leg=['Recall ({:.1f}%)'.format(recall*100),'Precision ({:.1f}%)'.format(precision*100),'Accuracy ({:.1f}%)'.format(Accuracy*100),'F1Score ({:.1f}%)'.format((F1score)*100)]
    Values=[recall,precision,Accuracy,F1score]
    p1=plt.bar(Labels,Values,width=0.5)
    plt.legend(p1,Labels_Leg,bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode(connected=True)
import pydotplus
import math
import itertools
import seaborn as sns


class ClassifierEvaluator:

    def calculate_metrics(self, model, model_name, validation_predictors, validation_classifications, plot=False,
                          model_predictions=None):

        if model_predictions is None:
            model_predictions = model.predict(validation_predictors)
        # model_score = model.score(validation_predictors, validation_classifications)
        cfn_matrix = confusion_matrix(validation_classifications, model_predictions)

        if plot is True:
            self._plot_confusion_matrix(cfn_matrix, classes=["Not Churned", "Churned"],
                                   title=model_name + " Confusion Matrix")

        # model_score = model.score(training_set_predictors, training_set_classifications)

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        precision = 0.0
        recall = 0.0
        F1 = 0.0
        MCC = 0.0

        indexes = range(len(model_predictions))
        for index in indexes:

            if model_predictions[index] == 1 and model_predictions[index] == validation_classifications[index]:
                true_positives += 1
            elif model_predictions[index] == 1 and model_predictions[index] != validation_classifications[index]:
                false_positives += 1
            elif model_predictions[index] == 0 and model_predictions[index] == validation_classifications[index]:
                true_negatives += 1
            elif model_predictions[index] == 0 and model_predictions[index] != validation_classifications[index]:
                false_negatives += 1

        if (true_positives + false_positives + true_negatives + false_negatives) == len(model_predictions) and (
                true_positives != 0 and true_negatives != 0):
            precision = true_positives / (true_positives + false_positives)

            recall = true_positives / (true_positives + false_negatives)

            F1 = (2 * precision * recall) / (precision + recall)

            MCC = ((true_positives * true_negatives) - (false_positives * false_negatives)) / math.sqrt(
                (true_positives + false_positives) * (true_positives + false_negatives) * (
                        true_negatives + false_negatives) * (true_negatives + false_positives))

        accuracy = (true_positives + true_negatives) / len(model_predictions)

        print(model_name + " Accuracy: ", accuracy)
        print(model_name + " Precision: ", str(precision))
        print(model_name + " Recall : ", str(recall))
        print(model_name + " FMeasure : ", str(F1))
        print(model_name + " MCC : ", str(MCC))
        print(model_name + " Confusion Matrix")
        print(cfn_matrix)

        return accuracy

    def _plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(title + ".png")


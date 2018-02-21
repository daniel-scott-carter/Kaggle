# import plotting libraries
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as offline
#offline.init_notebook_mode(connected=True)
import pydotplus
import math
import itertools
import seaborn as sns

class GraphsAndPlotsBuilder:

    variable = "I'm a variable"

    def get_feature_correlations(self, training_dataset):
        """
        :param training_dataset: the training dataset (with or without labels).
        :return:
        """
        sns.set()
        colourmap = plt.cm.plasma
        plt.figure(figsize=(24.0, 24.0))
        plt.title("Pearson Correlation of Features", y=1.0)

        ax = sns.heatmap(training_dataset.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colourmap,
                         linecolor="white", annot=True, cbar=True, xticklabels="auto", yticklabels="auto",
                         annot_kws={"size": 8})

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=0.975, bottom=0.15)

        plt.pause(0.05)
        plt.savefig("Plots/Feature Correlations.png")

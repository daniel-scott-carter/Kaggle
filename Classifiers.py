from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.externals.six import StringIO
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn import preprocessing

from mpl_toolkits.mplot3d import Axes3D
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

random_state = 2818
random_forest = None
GBM = None
plot_count = 0
top_ten_features = None
RF_predictions = None
one_weight = 0.0

def load_data():

    # Data.csv must exist!!!!!!!!!!!!!!!
    # If it does not: call GetAndCleanData.py first.
    df = pd.read_csv('Data.csv', infer_datetime_format=True, encoding="utf-8")
    print("Data Loaded from CSV!")

    return df


def load_last_3_day_data():
    three_day_df = pd.read_csv('3DaysData.csv', infer_datetime_format=True, encoding="utf-8")
    print("3Days Data Loaded from CSV!")

    return three_day_df


def save_last_3_day_predictions(dataframe_with_predictions):
    dataframe_with_predictions.to_csv("3DayPredictions.csv", index=False, encoding="utf-8")


def get_classifications(dataframe):

    # get the data as a numpy array
    classifications = list(dataframe["churnedWithin3Days"].values)
    return classifications


def get_features(dataframe):

    # get the dataframe as a numpy array
    print(str(list(dataframe)))

    dataframe = dataframe.drop(["churnedWithin3Days"], axis=1)
    features = dataframe.values

    print(features)
    return features


def calculate_weights(training_classifiers):

    global one_weight
    one_count = 0.0
    zero_count = 0.0
    #one_weight = 0.0

    for i in training_classifiers:
        if i == 1:
            one_count += 1.0
        if i == 0:
            zero_count += 1.0

    if one_count > zero_count:
        one_weight = one_count / zero_count

    if one_count < zero_count:
        one_weight = zero_count / one_count

    return one_weight


def get_feature_correlations(training_dataset):

    sns.set()
    colourmap = plt.cm.plasma
    plt.figure(figsize=(24.0, 24.0))
    plt.title("Pearson Correlation of Features", y=1.0)

    ax = sns.heatmap(training_dataset.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colourmap,
                        linecolor="white", annot=True, cbar=True, xticklabels="auto", yticklabels="auto", annot_kws={"size": 8})

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=0.975, bottom=0.15)

    plt.pause(0.05)
    plt.savefig("Feature Correlations.png")


def get_feature_importance(model, column_names):

    feature_importance = model.feature_importances_

    bar_data = [go.Bar(
        x=column_names,
        y=feature_importance,
        width=0.5,
        marker=dict(
            color=feature_importance,
            colorscale='Portland',
            showscale=True,
            reversescale=False
        ),
        opacity=0.6
    )]

    bar_layout = go.Layout(
        autosize=True,
        title=type(model).__name__ + ' Feature Importance',
        hovermode='closest',
        yaxis=dict(
            title='Feature Importance',
            ticklen=5,
            gridwidth=2
        ),
        showlegend=False
    )

    fig = go.Figure(data=bar_data, layout=bar_layout)
    offline.plot(fig, filename=type(model).__name__ + ' Feature Importance.html')


def get_average_feature_importance(list_of_models):

    importance_dataframe = pd.DataFrame()

    for model in list_of_models:

        importances = model.feature_importances_
        importance_dataframe[type(model).__name__] = importances

    mean_importances = importance_dataframe.mean(axis=1)

    bar_data = [go.Bar(
        x=importance_dataframe.columns,
        y=mean_importances,
        width=0.5,
        marker=dict(
            color=mean_importances,
            colorscale='Portland',
            showscale=True,
            reversescale=False
        ),
        opacity=0.6
    )]

    bar_layout = go.Layout(
        autosize=True,
        title='Avg Feature Importance',
        hovermode='closest',
        yaxis=dict(
            title='Feature Importance',
            ticklen=5,
            gridwidth=2
        ),
        showlegend=False
    )

    fig = go.Figure(data=bar_data, layout=bar_layout)
    offline.plot(fig, 'Avg Feature Importance')


def calculate_metrics(model, model_name, validation_predictors, validation_classifications):

    #model_predictions = model.predict(training_set_predictors)

    model_predictions = model.predict(validation_predictors)
    # model_score = model.score(validation_predictors, validation_classifications)
    cfn_matrix = confusion_matrix(validation_classifications, model_predictions)
    #_plot_confusion_matrix(cfn_matrix, classes=["Not Churned", "Churned"], title=model_name + " Confusion Matrix")
    #model_score = model.score(training_set_predictors, training_set_classifications)

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

    if (true_positives + false_positives + true_negatives + false_negatives) == len(model_predictions) and (true_positives != 0 and true_negatives != 0):
        precision = true_positives / (true_positives + false_positives)

        recall = true_positives / (true_positives + false_negatives)

        F1 = (2 * precision * recall) / (precision + recall)

        MCC = ((true_positives * true_negatives) - (false_positives * false_negatives)) / math.sqrt(
            (true_positives + false_positives) * (true_positives + false_negatives) * (
            true_negatives + false_negatives) * (true_negatives + false_positives))

    accuracy = (true_positives + true_negatives)/len(model_predictions)

    print(model_name + " Accuracy: ", accuracy)
    print(model_name + " Precision: ", str(precision))
    print(model_name + " Recall : ", str(recall))
    print(model_name + " FMeasure : ", str(F1))
    print(model_name + " MCC : ", str(MCC))
    print(model_name + " Confusion Matrix")
    print(cfn_matrix)


def train_random_forest(training_set, training_classifiers):
    start = time.time()
    global random_state
    global random_forest
    global one_weight
    global RF_predictions

    parameter_candidates = [{
        'n_estimators': [50, 100, 150, 200, 1000],
        'max_depth': [5, 10, 15, 20, None],
        'random_state': [random_state],
        'class_weight': [{1: one_weight}]
    }]

    # random_forest = RandomForestClassifier(n_estimators=1000,
    #                                        max_features="log2",
    #                                        class_weight={1: one_weight},
    #                                        random_state=random_state,
    #                                        n_jobs=-1)
    #
    # random_forest.fit(training_set, training_classifiers)

    grid = GridSearchCV(estimator=RandomForestClassifier(),
                        param_grid=parameter_candidates,
                        cv=5,
                        refit=True,
                        error_score=0,
                        n_jobs=-1)

    grid.fit(training_set, training_classifiers)

    print("Best parameters set found on development set:")
    print(grid.best_params_)
    print("Best Score Found: ")
    print(grid.best_score_)
    random_forest = grid.best_estimator_

    end = time.time()
    elapsed_time = end - start
    print("Time training RF: ", elapsed_time)

    return random_forest


def plot_RF_confusion_matrix(test_set_classifications):
    global RF_predictions
    global random_forest

    cfn_matrix = confusion_matrix(test_set_classifications, RF_predictions)
    _plot_confusion_matrix(cfn_matrix, classes=["Not Churned", "Churned"], title="Random Forest Confusion Matrix")


def _plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

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
    plt.savefig(title+".png")


def get_random_forest_feature_importance(training_set, columns):

    global random_forest
    global top_ten_features

    importances = random_forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in random_forest.estimators_], axis=0)
    indices = np.argsort(importances)

    print("Feature Ranking: ")

    for f in range(training_set.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))

    plt.figure(figsize=(19.0, 9.0))
    plt.title("Random Forest Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color="b", yerr=std[indices], align="center")
    plt.yticks(range(len(indices)), columns[indices])
    plt.xlabel('Relative Importance')
    plt.ion()
    plt.pause(0.05)
    plt.savefig("Random Forest Feature Importance.png")


def get_random_forest_graphs(columns, test_set, test_classifiers):
    global random_forest

    highest_index = 0
    highest_score = 0

    for index in range(len(random_forest.estimators_)):

        if random_forest.estimators_[index].score(test_set, test_classifiers) > highest_score:
            highest_score = random_forest.estimators_[index].score(test_set, test_classifiers)
            highest_index = index

    print("Index of Best Tree: ", highest_index)
    print("Accuracy of Best Tree: ", highest_score)

    dot_file = StringIO()
    tree_one = random_forest.estimators_[highest_index]
    tree.export_graphviz(tree_one,
                         out_file=dot_file,
                         feature_names=columns,
                         class_names=["Not Churned", "Churned"],
                         filled=True,
                         rounded=True,
                         special_characters=True)

    pydotplus.graph_from_dot_data(dot_file.getvalue()).write_png("Random_Forest_Tree_FULL.png")


def train_gradient_boost_machine(training_set, training_classifiers):
    global GBM
    global random_state
    start = time.time()

    # GBM = GradientBoostingClassifier(n_estimators=1000,
    #                                  learning_rate=0.1,
    #                                  loss="deviance",
    #                                  random_state=random_state)
    #
    # GBM.fit(training_set, training_classifiers)

    parameter_candidates = [{
        'loss': ["exponential"],
        'learning_rate': [0.1, 0.2, 0.3, 0.4],
        'n_estimators': [25, 50, 75, 100, 500],
        'max_depth': [3, 4, 5],
        'max_features': ["auto"],
        'random_state': [random_state],
    }]

    grid = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=parameter_candidates, cv=5, refit=True, error_score=0,
                        n_jobs=-1)

    grid.fit(training_set, training_classifiers)

    print("Best parameters set found on development set:")
    print(grid.best_params_)
    print("Best Score Found: ")
    print(grid.best_score_)

    GBM = grid.best_estim

    end = time.time()
    elapsed_time = end - start
    print("Time training GBM: ", elapsed_time)

    return GBM


def plot_partial_dependencies_one_way(columns, training_set_features):

    global GBM
    global top_ten_features

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 16.0
    fig_size[1] = 12.0
    plt.rcParams["figure.figsize"] = fig_size

    fig, axs = plot_partial_dependence(GBM,
                                       X=training_set_features,
                                       features=top_ten_features,
                                       feature_names=columns,
                                       n_jobs=1,
    grid_resolution = 50)

    fig.suptitle("Top 10 GBM Feature Dependencies")
    plt.subplots_adjust(top=0.9)
    plt.ion()
    plt.pause(0.05)
    plt.savefig("GBM Partial Dependencies.png")
    fig_size[0] = 8.0
    fig_size[1] = 6.0
    plt.rcParams["figure.figsize"] = fig_size


def plot_partial_dependencies_two_way(training_set, columns, index_feature_1, index_feature_2):

    global GBM

    fig = plt.figure()

    target_features = [index_feature_1, index_feature_2]

    pdp, axes = partial_dependence(GBM, target_features, X=training_set, grid_resolution=60)

    XX,YY = np.meshgrid(axes[0], axes[1])
    Z = pdp[0].reshape(list(map(np.size, axes))).T
    ax = Axes3D(fig)
    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu)
    ax.set_xlabel(columns[target_features[0]])
    ax.set_ylabel(columns[target_features[1]])
    ax.set_zlabel("Partial Dependence")
    ax.view_init(elev=22, azim=22)
    plt.colorbar(surf)
    plt.suptitle("Partial Dependence of " + columns[target_features[0]] + " and " + columns[target_features[1]] + " on Churned Value")
   # plt.subplots_adjust(top=0.9)
    plt.ion()
    plt.pause(0.05)
    plt.savefig("GBM_partial_two_way_dependency_"+str(index_feature_1)+str(index_feature_2)+".png")


def get_gradient_boost_machine_feature_importances(training_set, columns):
        global GBM
        global top_ten_features

        feature_importance = GBM.feature_importances_
        # feature_importance = (feature_importance/feature_importance.max())

        indices = np.argsort(feature_importance)
        top_ten_features = indices[15:30]

        for f in range(training_set.shape[1]):
            print("%d. feature %s (%f)" % (f + 1, columns[indices[f]], feature_importance[indices[f]]))

        plt.figure(figsize=(19.0, 9.0))
        plt.title("Gradient Boost Feature Importances")
        plt.barh(range(len(indices)), feature_importance[indices], color="b",  align="center")
        plt.yticks(range(len(indices)), columns[indices])
        plt.xlabel('Relative Importance')
        plt.ion()
        plt.pause(0.05)
        plt.savefig("GBM Feature Importance.png", pad_inches=4)


def train_artificial_neural_network(training_set, training_classifications):
    global one_weight
    global random_state
    training_set = preprocessing.normalize(training_set)
    start = time.time()

    # neural_net = MLPClassifier(hidden_layer_sizes=1000,
    #                            random_state=random_state)
    # neural_net.fit(training_set, training_classifications)

    parameter_candidates = [{
        'activation': ["identity", "logistic", "tanh", "relu"],
        'solver': ["lbfgs", "adam"],
        'learning_rate': ["costant", "invscaling", "adaptive"],
        'random_state': [random_state],
    }]

    grid = GridSearchCV(estimator=MLPClassifier(), param_grid=parameter_candidates, cv=5, refit=True, error_score=0,
                        n_jobs=-1)

    grid.fit(training_set, training_classifications)
    end = time.time()
    elapsed_time = end - start
    print("Time training ANN: ", elapsed_time)
    print("Best parameters set found on development set:")
    print(grid.best_params_)
    print("Best Score Found: ")
    print(grid.best_score_)

    neural_net = grid.best_estimator_

    return neural_net


def train_naive_bayes(training_set, training_classifications):
    global one_weight

    #training_set = preprocessing.normalize(training_set)
    #test_set = preprocessing.normalize(test_set)

    start = time.time()
    naive_bayes = GaussianNB()

    naive_bayes.fit(training_set, training_classifications)
    end = time.time()
    elapsed_time = end - start

    print("Time training NB: ", elapsed_time)

    return naive_bayes


def train_k_nearest_neighbours(training_set, training_classifications):
    global one_weight


    training_set = preprocessing.normalize(training_set)

    start = time.time()
    # nearest_neighbours = KNeighborsClassifier(weights="distance", n_jobs=-1)
    #
    # nearest_neighbours.fit(training_set, training_classifications)

    parameter_candidates = [{
        'n_neighbors': [4, 5, 7, 10, 12],
        'weights': ["uniform", "distance"],
        'algorithm': ["auto", "ball_tree", "kd_tree", "brute"],
        'leaf_size': [10, 20, 30, 40, None],
        'p': [1, 2]
    }]

    grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameter_candidates, cv=5, refit=True, error_score=0,
                        n_jobs=-1)

    grid.fit(training_set, training_classifications)
    end = time.time()
    elapsed_time = end - start
    print("Time training KNN: ", elapsed_time)
    print("Best parameters set found on development set:")
    print(grid.best_params_)
    print("Best Score Found: ")
    print(grid.best_score_)

    nearest_neighbours = grid.best_estimator_
    return nearest_neighbours


def train_extremely_random_forest(training_set, training_classifications):
    global one_weight
    global random_state

    start = time.time()
    # xtreme_random_forest = ExtraTreesClassifier(n_estimators=2000,
    #                                             max_features="log2",
    #                                             class_weight={1: one_weight},
    #                                             random_state=random_state,
    #                                             n_jobs=-1)

    parameter_candidates = [{
        'n_estimators': [50, 100, 150, 200],
        'criterion': ["gini", "entropy"],
        'n_estimators': [25, 50, 75, 100],
        'max_features': ["auto", "log2", None],
        'max_depth': [5, 10, 15, 20, None],
        'random_state': [random_state],
        'class_weight': [{1: one_weight}]
    }]

    grid = GridSearchCV(estimator=ExtraTreesClassifier(), param_grid=parameter_candidates, cv=5, refit=True, error_score=0,
                        n_jobs=-1)

    grid.fit(training_set, training_classifications)
    end = time.time()
    elapsed_time = end - start
    print("Time training XRF: ", elapsed_time)
    print("Best parameters set found on development set:")
    print(grid.best_params_)
    print("Best Score Found: ")
    print(grid.best_score_)



    xtreme_random_forest = grid.best_estimator_
    return xtreme_random_forest


def train_adaboost(training_set, training_classifications):
    global one_weight
    global random_state

    start = time.time()
    # adaboost = AdaBoostClassifier(n_estimators=2000,
    #                               random_state=random_state)
    #
    # adaboost.fit(training_set, training_classifications)

    parameter_candidates = [{
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.5, 1, 1.5, 2],
        'n_estimators': [25, 50, 75, 100],
        'algorithm': ["SAMME", "SAMME.R"],
        'random_state': [random_state],
    }]

    grid = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=parameter_candidates, cv=5, refit=True, error_score=0,
                        n_jobs=-1)

    grid.fit(training_set, training_classifications)
    end = time.time()
    elapsed_time = end - start
    print("Time training AdaBoost: ", elapsed_time)
    print("Best parameters set found on development set:")
    print(grid.best_params_)
    print("Best Score Found: ")
    print(grid.best_score_)

    #xtreme_random_forest = grid.best_estimator_

    end = time.time()
    elapsed_time = end - start

    print("Time training AdaBoost: ", elapsed_time)

    adaboost = grid.best_estimator_
    return adaboost


def train_stochastic_gradient_descent(training_set, training_classifications):
    global one_weight
    global random_state

    standard = preprocessing.StandardScaler()
    training_set = standard.fit_transform(training_set)

    start = time.time()
    # sgd = SGDClassifier(loss="log",
    #                     class_weight={1: one_weight},
    #                     random_state=random_state,
    #                     n_jobs=-1)
    #
    # sgd.fit(training_set, training_classifications)

    parameter_candidates = [{
        'loss': ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        'penalty': ["l1", "l2", "elasticnet", "none"],
        "random_state": [random_state],
        "learning_rate": ["constant", "optimal", "invscaling"],
        "eta0": [0.00001, 0.0001, 0.001, 0.01],
        "class_weight": [{1: one_weight}]
    }]

    grid = GridSearchCV(estimator=SGDClassifier(), param_grid=parameter_candidates, cv=5, refit=True, error_score=0,
                        n_jobs=-1)

    grid.fit(training_set, training_classifications)
    end = time.time()
    elapsed_time = end - start
    print("Time training SGD: ", elapsed_time)
    print("Best parameters set found on development set:")
    print(grid.best_params_)
    print("Best Score Found: ")
    print(grid.best_score_)

    sgd = grid.best_estimator_
    return sgd


def train_quadratic_discriminant_analysis(training_set, training_classifications):
    global one_weight

    #training_set = preprocessing.normalize(training_set)
    #test_set = preprocessing.normalize(test_set)

    start = time.time()
    QDA = QuadraticDiscriminantAnalysis()

    QDA.fit(training_set, training_classifications)
    end = time.time()
    elapsed_time = end - start

    print("Time training Quad Discriminant Analysis: ", elapsed_time)

    return QDA


def done_plotting():
    while True:
        plt.pause(0.05)

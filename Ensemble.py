import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


predictions_df = pd.DataFrame()


def get_predictions_dataframe():
    global predictions_df
    return predictions_df


def get_model_correlations():
    global predictions_df

    sns.set()
    colourmap = plt.cm.plasma
    plt.figure(figsize=(24.0, 24.0))
    plt.title("Pearson Correlation of Model Predictions", y=1.0)

    ax = sns.heatmap(predictions_df.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colourmap,
                     linecolor="white", annot=True, cbar=True, xticklabels="auto", yticklabels="auto",
                     annot_kws={"size": 8})

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=0.975, bottom=0.15)

    plt.pause(0.05)
    plt.savefig("Model Correlations.png")


def get_ensemble_predict_fitness(list_of_models, test_set, test_classifier):
    global predictions_df

    model_num = len(list_of_models)

    if model_num == 0:
        return 0

    start = time.time()
    ensemble_predictions = []
    correct_predictions = []

    normal_names = ["SVC"]

    counter = 0
    predictions = []
    model_scores = []
    best_score = 0
    best_index = 0
    # precisions = []
    # recalls = []


    best_weight = 1

    # if model_num == 1 or model_num == 2 or model_num == 3 or model_num == 4:
    #     best_weight = 1
    # else:
    #     best_weight = model_num - 2

    for classifier in list_of_models:

        # if type(classifier).__name__ in normal_names:
        #     testing_set = preprocessing.normalize(test_set)
        # else:
        testing_set = test_set

        model_pred = classifier.predict(testing_set)

        predictions_df[type(classifier).__name__ ] = model_pred
        predictions.append(model_pred)
        classifier_score = classifier.score(testing_set, test_classifier)
        model_scores.append([classifier_score])

        if classifier_score > best_score:
            best_score = classifier_score
            best_index = counter

        counter += 1

    iters = range(len(predictions[0]))

    for i in iters:

        # get each prediction probab for each model.
        zero_preds = []
        one_preds = []
        total_preds = 0
        top_pred = 0

        for j in range(model_num):
            # model_index = sorted_model_scores[j][0]
            class_probs = predictions[j][i]

            '''Rank Weighted Ensemble'''
            # if j == best_index:
            #     top_pred = predictions[model_index][i]
            #
            # if class_probs == 0:
            #     zero_preds.extend([class_probs] * (model_num - j))
            # else:
            #     one_preds.extend([class_probs] * (model_num - j))

            '''Best vs Rest Ensemble'''
            if j == best_index:
                if class_probs == 0:
                    zero_preds.extend([class_probs] * best_weight)
                else:
                    one_preds.extend([class_probs] * best_weight)
            else:
                if class_probs == 0:
                    zero_preds.extend([class_probs])
                else:
                    one_preds.extend([class_probs])

            total_preds += (model_num-j)

        # will have to get probabs for all models here per prediction, then average by dividing by number of models

        if len(one_preds) == len(zero_preds):
            if top_pred == 1:
            #if avg_prec_or_fpr < avg_rec_or_fnr:
                ensemble_predictions.append(1)
                if test_classifier[i] == 1:
                    correct_predictions.append(1)
                else:
                    correct_predictions.append(0)
            else:
                ensemble_predictions.append(0)
                if test_classifier[i] == 0:
                    correct_predictions.append(1)
                else:
                    correct_predictions.append(0)
        elif len(one_preds) > len(zero_preds):
            ensemble_predictions.append(1)
            if test_classifier[i] == 1:
                correct_predictions.append(1)
            else:
                correct_predictions.append(0)
        else:
            ensemble_predictions.append(0)
            if test_classifier[i] == 0:
                correct_predictions.append(1)
            else:
                correct_predictions.append(0)

    end = time.time()
    elapsed_time = end - start
    print("Time evaluating: ", elapsed_time)
    print("Total Predictions = ", len(predictions[0]))
    print("Correct Predictions = ", sum(correct_predictions))
    accuracy = float(float(sum(correct_predictions)) / float(len(ensemble_predictions)))
    print("Ensemble Accuracy = ", accuracy)

    return accuracy


def get_ensemble_predictions(list_of_models, dataset_to_predict, test_set_features, test_set_classifications):
    model_num = len(list_of_models)

    if model_num == 0:
        return 0

    ensemble_predictions = []
    correct_predictions = []

    normal_names = ["SVC"]

    counter = 0
    predictions = []
    model_scores = []
    best_score = 0
    best_index = 0
    # precisions = []
    # recalls = []


    # best_weight = 1

    if model_num == 1 or model_num == 2 or model_num == 3 or model_num == 4:
        best_weight = 1
    else:
        best_weight = model_num - 2

    for classifier in list_of_models:

        testing_set = test_set_features

        model_pred = classifier.predict(dataset_to_predict)
        predictions.append(model_pred)

        classifier_score = classifier.score(testing_set, test_set_classifications)
        model_scores.append([classifier_score])

        if classifier_score > best_score:
            best_score = classifier_score
            best_index = counter

        counter += 1

    iters = range(len(predictions[0]))

    for i in iters:

        # get each prediction probab for each model.
        zero_preds = []
        one_preds = []
        total_preds = 0
        top_pred = 0

        for j in range(model_num):
            # model_index = sorted_model_scores[j][0]
            class_prediction = predictions[j][i]

            '''Rank Weighted Ensemble'''
            # if j == best_index:
            #     top_pred = predictions[model_index][i]
            #
            # if class_probs == 0:
            #     zero_preds.extend([class_probs] * (model_num - j))
            # else:
            #     one_preds.extend([class_probs] * (model_num - j))

            '''Best vs Rest Ensemble'''
            if j == best_index:
                if class_prediction == 0:
                    zero_preds.extend([class_prediction] * best_weight)
                else:
                    one_preds.extend([class_prediction] * best_weight)
            else:
                if class_prediction == 0:
                    zero_preds.extend([class_prediction])
                else:
                    one_preds.extend([class_prediction])

            total_preds += (model_num - j)

        # will have to get probabs for all models here per prediction, then average by dividing by number of models

        if len(one_preds) == len(zero_preds):
            if top_pred == 1:
                # if avg_prec_or_fpr < avg_rec_or_fnr:
                ensemble_predictions.append(1)
            else:
                ensemble_predictions.append(0)

        elif len(one_preds) > len(zero_preds):
            ensemble_predictions.append(1)
        else:
            ensemble_predictions.append(0)

    print("Total Predictions = ", len(predictions[0]))
    print("What have I done...")
    return ensemble_predictions

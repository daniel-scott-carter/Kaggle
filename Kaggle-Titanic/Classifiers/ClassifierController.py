import numpy as np
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------------
# Import Classifiers.py
import Classifiers as cl
import Ensemble as en
from Ensemble import GAEnsembleSelection

scaler = StandardScaler()
df = cl.load_data()

df["battlesLostPvE"] = df["battlesLostPvE"].loc[df["battlesLostPvE"].notnull()].astype(int)
df["pointsConquest"] = df["pointsConquest"].loc[df["pointsConquest"].notnull()].astype(int)

print(df.dtypes)

msk = np.random.rand(len(df)) < 0.75

training_set = df[msk]
test_set = df[~msk]

validation_msk = np.random.rand(len(test_set)) < 0.70

validation_set = test_set[validation_msk]
test_set = test_set[~validation_msk]

print("Length of df: ", len(df))
print("Length of train: ", len(training_set))
print("Length of test: ", len(test_set))

columns = training_set.drop(["churnedWithin3Days"], axis=1).columns

cl.get_feature_correlations(training_set)
full_set_classifications = cl.get_classifications(df)

training_set_features = cl.get_features(training_set)
training_set_classifications = cl.get_classifications(training_set)
cl.calculate_weights(training_set_classifications)

validation_set_features = cl.get_features(validation_set)
validation_set_classifications = cl.get_classifications(validation_set)

test_set_features = cl.get_features(test_set)
test_set_classifications = cl.get_classifications(test_set)

models = []

random_forest = cl.train_random_forest(training_set_features, training_set_classifications)
# cl.get_feature_importance(random_forest, columns)
cl.calculate_metrics(random_forest, "Random Forest", validation_set_features, validation_set_classifications)
models.append(random_forest)
# rf_importances = cl.get_random_forest_feature_importance(training_set_features, columns)

# cl.plot_RF_confusion_matrix(test_set_classifications)

'''
cl.get_random_forest_graphs(columns,
                            test_set_features,
                            test_set_classifications)
'''


gradient_boost_machine = cl.train_gradient_boost_machine(training_set_features,
                                                         training_set_classifications)
cl.calculate_metrics(gradient_boost_machine, "Gradient Boost", validation_set_features, validation_set_classifications)
models.append(gradient_boost_machine)

# cl.get_gradient_boost_machine_feature_importances(training_set_features, columns)

'''
cl.plot_partial_dependencies_one_way(columns,
                                     training_set_features)
'''
'''
cl.plot_partial_dependencies_two_way(training_set_features,
                                   columns,
                                   29,
                                   28)
'''


neural_net = cl.train_artificial_neural_network(training_set_features, training_set_classifications)
cl.calculate_metrics(neural_net, "Neural Net", validation_set_features, validation_set_classifications)
models.append(neural_net)

naive_bayes = cl.train_naive_bayes(training_set_features, training_set_classifications)
cl.calculate_metrics(naive_bayes, "Naive Bayes", validation_set_features, validation_set_classifications)
models.append(naive_bayes)


k_nearest_neighbour = cl.train_k_nearest_neighbours(training_set_features, training_set_classifications)
cl.calculate_metrics(k_nearest_neighbour, "KNN", validation_set_features, validation_set_classifications)
models.append(k_nearest_neighbour)


xtreme_forest = cl.train_extremely_random_forest(training_set_features, training_set_classifications)
cl.calculate_metrics(xtreme_forest, "Xtreme Random Forest", validation_set_features, validation_set_classifications)
models.append(xtreme_forest)


adaboost = cl.train_adaboost(training_set_features, training_set_classifications)
cl.calculate_metrics(adaboost, "AdaBoost", validation_set_features, validation_set_classifications)
models.append(adaboost)


#
stochastic_gradient_descent = cl.train_stochastic_gradient_descent(training_set_features, training_set_classifications)
cl.calculate_metrics(stochastic_gradient_descent, "SGD", validation_set_features, validation_set_classifications)
models.append(stochastic_gradient_descent)

quadratic_discriminant = cl.train_quadratic_discriminant_analysis(training_set_features, training_set_classifications)
cl.calculate_metrics(quadratic_discriminant, "QDA", validation_set_features, validation_set_classifications)
models.append(quadratic_discriminant)

en.get_ensemble_predict_fitness(models, validation_set_features, validation_set_classifications)

en.get_model_correlations()

GAEnsembleSelection.setup(models, 7, 50, validation_set_features, validation_set_classifications)
best_model_sublist = GAEnsembleSelection.execute()


# test our GA Ensemble on the TEST SET here, for final evaluation

 # cl.get_average_feature_importance(best_model_sublist)

prediction_dframe = cl.load_last_3_day_data()

predictions = en.get_ensemble_predictions(best_model_sublist, prediction_dframe, test_set_features, test_set_classifications)
prediction_dframe["Churn_Prediction"] = predictions
cl.save_last_3_day_predictions(prediction_dframe)


cl.done_plotting()

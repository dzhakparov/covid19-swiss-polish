import numpy as np

# Classifiers
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

0
# CLASSIFIERS

VotingClassifier_model = True
VotingClassifier_params = {'voting': 'hard',
                          # 'weights': [2,1]
                           }

use_classifiers = ['AdaBoost', 'Bagging', 'Random Forest', 'Ridge', 'KNN']


classifiers_collection = {
    "AdaBoost": AdaBoostClassifier(),
    "Bagging": BaggingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Ridge": RidgeClassifier(),
    "SGD": SGDClassifier(),
    "MLP": MLPClassifier(),
    "Extra Trees Ensemble": ExtraTreesClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(),
    "DTC": DecisionTreeClassifier(),
    "ETC": ExtraTreeClassifier(),
    "SVC": SVC(),
    "LSVC": LinearSVC(),
    "MLP": MLPClassifier()
}

parameters_collection = {
    "AdaBoost": {
        "classifier__base_estimator": [DecisionTreeClassifier(max_depth=ii) for ii in range(1, 6)],
        "classifier__n_estimators": [200],
        "classifier__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 1.0]
    },
    "Bagging": {
        "classifier__base_estimator": [DecisionTreeClassifier(max_depth=ii) for ii in range(1, 6)],
        "classifier__n_estimators": [200],
        "classifier__max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__n_jobs": [-1]
    },
    "Extra Trees Ensemble": {
        "classifier__n_estimators": [200],
        "classifier__class_weight": [None, "balanced"],
        "classifier__max_features": ["auto", "sqrt", "log2"],
        "classifier__max_depth": [3, 4, 5, 6, 7, 8],
        "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
        "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
        "classifier__criterion": ["gini", "entropy"],
        "classifier__n_jobs": [-1]
    },
    "Gradient Boosting": {
        "classifier__learning_rate": [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
        "classifier__n_estimators": [200],
        "classifier__max_depth": [2, 3, 4, 5, 6],
        "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
        "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
        "classifier__max_features": ["auto", "sqrt", "log2"],
        "classifier__subsample": [0.8, 0.9, 1]
    },
    "Random Forest": {
        "classifier__n_estimators": [200],
        # "classifier__class_weight": [None, "balanced"],
        # "classifier__max_features": ["auto", "sqrt", "log2"],
        "classifier__max_features": ["auto"],
        # "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
        "classifier__max_depth": [3, 5, 7],
        "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
        # "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
        "classifier__criterion": ["gini", "entropy"],
        "classifier__n_jobs": [-1]
    },
    "Ridge": {
        "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
    },
    "SGD": {
        "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
        "classifier__penalty": ["l1", "l2"],
        "classifier__n_jobs": [-1]
    },
    "BNB": {
        "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
    },
    "KNN": {
        "classifier__n_neighbors": list(range(1, 20)),
        "classifier__p": [1, 2, 3, 4, 5],
        "classifier__leaf_size": [5, 10, 15],  # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "classifier__n_jobs": [-1]
    },
    "SVC": {
        "classifier__kernel": ["linear", "rbf", "poly"],
        "classifier__gamma": ["auto"],
        "classifier__C": [0.1, 0.5, 1, 5, 10, 50, 100],
        "classifier__degree": [1, 2, 3, 4, 5, 6]
    },
    "LSVC": {
        "classifier__penalty": ["l2"],
        "classifier__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
    },
    "MLP": {
        "classifier__hidden_layer_sizes": [(5), (10), (5, 5), (10, 10), (5, 5, 5), (10, 10, 10)],
        "classifier__activation": ["identity", "logistic", "tanh", "relu"],
        "classifier__learning_rate": ["constant", "invscaling", "adaptive"],
        "classifier__max_iter": [100, 200, 300, 500, 1000, 2000],
        "classifier__alpha": list(10.0 ** -np.arange(1, 10)),
    }
}

# Create list of tuples with classifier label and classifier object
classifiers = {}
parameters = {}

for classifier in use_classifiers:
    classifiers.update({classifier: classifiers_collection.get(classifier)})
    parameters.update({classifier: parameters_collection.get(classifier)})
import numpy as np
from config import general, modeling
from config_classifier import parameters, classifiers, VotingClassifier_model, VotingClassifier_params
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from joblib import dump
from src.helpers import log, get_train_test_files
from sklearn.ensemble import VotingClassifier


@log()
def run():

    # script = str.split(os.path.basename(__file__), '.')[0]
    # logging, console = setup(script)
    # logging.info(f"parameters from 'ml_params' ('config.py'):\n\n {pformat(ml_params)}\n")

    X_train_ml, X_test_ml, y_train, y_test = get_train_test_files(['X_train_ml', 'X_test_ml', 'y_train', 'y_test'],
                                                                  error_message= f"file not found! Run 'ml_pipeline_sax' first to create train- and test-files and "
                                                                  f"'ml_pipeline_transformation' to generate 'X_train_ml' and 'X_test_ml'"
                                                                  )

    models = Models(classifiers=classifiers, hyperparameters=parameters, **modeling)
    models.fit(X_train_ml, y_train)
    fitted_models = models.results

    # build VotingClassifier of best models
    if VotingClassifier_model:
        if 'weights' not in VotingClassifier_params.keys() or \
                ('weights' in VotingClassifier_params.keys() and
                len(VotingClassifier_params['weights']) !=len(fitted_models)):
            VotingClassifier_params.update({'weights': [1] * (len(fitted_models))})

        estimators = [(i, fitted_models.get(i).best_estimator_.named_steps.classifier) for i in list(fitted_models.keys())]

        eclf = VotingClassifier(estimators = estimators, **VotingClassifier_params)
        eclf = eclf.fit(X_train_ml, y_train)
        fitted_models.update({'VotingClassifier': eclf})

    dump(fitted_models, f"{general['output_path']}fitted_models.joblib")

    # logging.info(f"\ncalculated new classification models and stored fitted models in "
    #              f"{general['output_path']}fitted_models.joblib.\n")

    # # logging infos to models
    # used_classifiers = [item for item in fitted_models.keys()]
    # # logging.info(f"\n\nused classifiers for modeling: \n{pformat(used_classifiers)}\n")
    #
    # used_parameters = {item: fitted_models[item].param_grid for item in parameters if item in used_classifiers}
    # # logging.info(f"\n\nused parameters for modeling: \n{pformat(used_parameters)}\n")
    #
    # best_models = {item: fitted_models[item].best_params_ for item in fitted_models}
    # # logging.info(f"\n\nbest models:\n{pformat(best_models)}\n")
    #
    # best_scores = {item: fitted_models[item].best_score_ for item in fitted_models}
    # # logging.info(f"\n\nbest score (mean cross-validation):\n{pformat(best_scores)}\n")


class Models:

    def __init__(self, classifiers, hyperparameters, logging=None, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc',
                 **kwargs):
        """
        :param classifiers: dictionary of desired classifiers (exp: {'AdaBoost': AdaBoostClassifier(), ... }
        :param hyperparameters:
        """
        self.classifiers = classifiers
        self.hyperparameters = hyperparameters
        self.logging = logging
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.verbose = verbose
        self.results = {}

    def fit(self, X_train, y_train):
        for classifier_label, classifier in classifiers.items():
            steps = [("classifier", classifier)]
            pipeline = Pipeline(steps=steps)  # here an other preprocess-step could be implemented
            param_grid = parameters[classifier_label]
            self.gscv = GridSearchCV(pipeline, param_grid, cv=self.cv, n_jobs=self.n_jobs, verbose=self.verbose,
                                     scoring=self.scoring)
            self.gscv.fit(X_train, np.ravel(y_train))

            self._build_results(classifier_label)

    def _build_results(self, classifier_label):
        self.results.update({classifier_label: self.gscv})


if __name__ == '__main__':
    pass
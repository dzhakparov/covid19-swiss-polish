import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin

""" Following functions are made to fit several estimators at the same time
and perform the GridSearchCV to select the best performing model """

class EstimatorSelectionHelper:

    """
    Estimator Helper to fit and gread search several models with several different hyperparameters at once in one pipeline

    Parameters
    ----------
    random_state
    n_datasets
    iterations
    verbose


    Returns
    ----------
    dataframe

    """

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, n_jobs=-1, n_splits = 10, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=StratifiedKFold(n_splits = n_splits), n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='median_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'median_score':np.median(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(10):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'median_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

class MiceImputer(BaseEstimator, TransformerMixin):

    """
    Class used to impute missing values with miceforest imputer

    Parameters
    ----------
    random_state
    n_datasets
    iterations
    verbose


    Returns
    ----------
    dataframe

    """

    def __init__(self, random_state = 999, n_datasets = 10, variable_schema = None, iterations = 5, verbose = False):
        self.random_state = random_state
        self.n_datasets = n_datasets
        self.iterations = iterations
        self.verbose = verbose
        self.variable_schema = variable_schema

    def fit(self, X, y = None):
        import miceforest as mf
        self.complete_data_list = []

        self.kernel = mf.MultipleImputedKernel(
        data = X,
        datasets = self.n_datasets,
        save_all_iterations = True,
        variable_schema = self.variable_schema,
        random_state = self.random_state)

        self.kernel.mice(iterations = self.iterations, n_jobs = -1, verbose = self.verbose)

        for i in range(0,self.n_datasets):
            self.df = self.kernel.complete_data(i)
            self.complete_data_list.append(self.df)
            self.output_df_grouping = pd.concat(self.complete_data_list, axis = 0)
            self.output_mean = self.output_df_grouping.groupby(level = 0).mean()

        return self

    def transform(self, X, y = None):
        X = self.output_mean
        #print(f'df contains {sum(pd.isnull(X))} missing values.')
        print('Missing values present? ', X.isnull().values.any())
        # add a way to check if there any NaNs left in the dataframe
        return X

    def impute_new_data(self, X, y = None):
        self.new_data_list = []
        new_data = self.kernel.impute_new_data(X)

        for i in range(0,self.n_datasets):
            #self.new_df = new_data.complete_data(i)
            self.new_data_list.append(new_data.complete_data(i))
            self.new_df_grouping = pd.concat(self.new_data_list, axis = 0)
            self.output_mean_new = self.new_df_grouping.groupby(level = 0).mean()

            new_data_out = self.output_mean_new

        return new_data_out

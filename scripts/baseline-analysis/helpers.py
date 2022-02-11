import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.utils import estimator_html_repr
from pathlib import Path
from pprint import pprint, pformat

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


class ModelEvaluation:

    def __init__(self, working_file, target, simulation_file, pipe):
        """ Evaluates a .csv simulation-file
        :param working_file: path to pickled pandas DataFrame with raw values as string
        :param target: target feature as string
        :param simulation_file: path to .csv-file with simulation results as string
        :param pipe: path to .joblib-file of used sklearn Pipeline as string
        """
        self.working_file = working_file
        self.target = target
        self.simulation_file = simulation_file
        self.pipe = pipe
        self.X = None
        self.y = None
        self.best_algorithm = None
        self.best_param = None
        self.stats = None
        self.preprocessed_x = None
        self.df_feature_importance = None
        self.model = None
        self._read_in_data()

    @staticmethod
    def _path_conversion(path):
        try:
            path = Path(path)
        except TypeError:
            raise Exception(f"conversion from '{path}' to Path-object didn't work!")
        return path

    def _validate_path(self, path):
        path = self._path_conversion(path)

        if not path.exists():
            raise Exception(f"file '{path}' does not exists")
        return path

    def _read_in_data(self):
        try:
            self.working_file = pd.read_pickle(self.working_file)
            self.simulation_file = pd.read_csv(self.simulation_file, index_col=0)
            self.pipe = joblib.load(self.pipe)
            self.X = self.working_file.drop([self.target], axis=1)
            self.y = self.working_file[self.target]
        except FileNotFoundError:
            raise Exception(f"file not found...")

    def chose_model(self, test_size, model='best', sorted_by=('test_accuracy', 'mean_cv_accuracy')):
        """
        :param test_size: test_size to be evaluated as float
        :param model: 'best' (default) or name of model to be analyzed
        :param sorted_by: tuple with name of columns to evaluated as best model (sort will be in ascending order ->
        higher values are better)
        """

        data_red = self.simulation_file[self.simulation_file['test_size'] == test_size]
        best_algorithms = data_red.groupby("algorithm").mean().sort_values(by=list(sorted_by), ascending=False)

        if model == 'best':
            self.best_algorithm = best_algorithms.index[0]
        else:
            self.best_algorithm = model

        best_params = data_red[data_red['algorithm'] == self.best_algorithm].groupby('best_params').count().sort_values(
            by="algorithm", ascending=False)
        self.best_param = self._build_best_param_from_str(best_params)

        self.stats = data_red.loc[:, ['mean_cv_accuracy', 'mean_train_accuracy', 'test_accuracy']][
            data_red['algorithm'] == self.best_algorithm].agg(['mean', 'std'])

    @staticmethod
    def _build_best_param_from_str(params):
        best_param = {}
        param_str = params.index[0]
        param_str = param_str[1:-1]
        param_str_list = str.split(param_str, ',')
        for item in param_str_list:
            item_list = str.split(item, ':')
            for ch in ['"', '\\', "'", " "]:
                item_list[0] = item_list[0].replace(ch, '')
            best_param.update({item_list[0]: eval(item_list[1])})
        return best_param

    def _build_model(self):

        updated_params = self.best_param
        updated_params = {key: [value] for key, value in updated_params.items()}  # values have to be defined as list
        updated_pipe = self.pipe
        updated_pipe.steps.pop(-1)  # remove estimator (none)
        updated_pipe.steps.append(('estimator', eval(f"{self.best_algorithm}()")))  # add estimator

        # additional step to update pipe with params (only best set)! -> model
        kfold = KFold(n_splits=2, shuffle=True)
        self.model = GridSearchCV(estimator=updated_pipe, param_grid=updated_params, cv=kfold, n_jobs=-1,
                                  return_train_score=True, scoring='accuracy', refit=True)
        self.model.fit(self.X, self.y)

    def calculate_feature_importance(self, no_features=5):
        """
        calculate feature importance with SFS
        :return:
        """
        self._build_model()

        position_estimator = [idx for idx, i in enumerate(list(self.model.best_estimator_.named_steps))
                              if i == 'estimator'][0]
        best_estimator = self.model.best_estimator_.steps.pop(position_estimator)[1]
        self.preprocessed_x = self.model.best_estimator_.fit_transform(self.X)

        if isinstance(self.preprocessed_x, pd.DataFrame):
            cols = self.preprocessed_x.columns
        else:
            cols = self.X.columns

        # apply feature importance via SFS
        sfs = SFS(best_estimator,
                  k_features=no_features,
                  forward=True,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1)

        sfs = sfs.fit(self.preprocessed_x, self.y, custom_feature_names=cols.to_list())
        self.df_feature_importance = pd.DataFrame.from_dict(sfs.get_metric_dict()).T

    def _transform_feature_importance_to_list(self):
        feature_list = []
        for idx, item in enumerate(self.df_feature_importance['feature_names']):
            if idx == 0:
                feature_list.append(item[0])
            else:
                diff = [i for i in item if i not in feature_list]
                feature_list.append(diff[0])
        return feature_list

    def plot_scoring_overview(self, store_path=None):
        """
        :param store_path: None (default) or valid path to store plot. If path is not NOne the plot will not be returned
        :return: fig object (store = None) or None (store != None)
        """
        data_mod = self.simulation_file.melt(id_vars=['algorithm', 'test_size', 'random_state'],
                                             value_vars=['mean_cv_accuracy', 'mean_train_accuracy', 'test_accuracy'])
        # build plot
        sns.set_style("whitegrid")
        sns.catplot(data=data_mod, x='test_size', y='value', col='algorithm', hue='variable', kind='box', col_wrap=2)
        if store_path is not None:
            plt.savefig(f"{store_path}/train_test_analysis.png")
        else:
            plt.show()

    def plot_pipe(self, type='pipe', store_path=None):
        """ plots a html representation of the pipeline (or best_model)
        :param type: 'pipe' (default) or 'best_model'
        :param store_path: path to store output as string
        """
        if store_path is not None:
            path = self._validate_path(store_path)
        else:
            path = Path(os.getcwd())

        if type == 'pipe':
            with open(f"{path}/plot_pipe.html", 'w') as f:
                f.write(estimator_html_repr(self.pipe))
        else:
            if self.model is None:
                self._build_model()
            with open(f"{path}/plot_best_model.html", 'w') as f:
                f.write(estimator_html_repr(self.model))

    def get_stats_model(self):
        return self.stats

    def get_algorithm(self):
        return self.best_algorithm

    def get_best_param(self):
        return self.best_param

    def get_most_important_features(self, type='list'):
        """
        get most important features as pandas DataFrame or as list
        :param type: 'list' (default) or 'df'
        :return: list or pandas DataFrame of m,ost important features selected bei SFS with according algorithm
        """
        if self.df_feature_importance is not None:
            if type == 'list':
                return self._transform_feature_importance_to_list()
            else:
                return self.df_feature_importance

    # TODO
    def _get_all_models(self):
        return ['LogisticRegression', 'KNeighborsClassifier', 'RandomForestClassifier',
                'GradientBoostingClassifier', 'SVC', 'AdaBoostClassifier', 'BaggingClassifier']

    def analyze_models(self, models=None, test_size=0.25, no_features=None, **kwargs):
        """

        :param models:
        :return:
        """
        if models is None:
            models = self._get_all_models()

        overview = []
        for model in models:
            model_stats = {}
            self.chose_model(test_size=test_size, model=model, **kwargs)
            model_stats.update({'1_name': self.get_algorithm()})
            model_stats.update({'3_stats': self.get_stats_model()})
            model_stats.update({'2_best_param': self.get_best_param()})
            if no_features is not None:
                self.calculate_feature_importance(no_features=no_features)
                model_stats.update({'4_features': self.get_most_important_features(type='list')})
            overview.append(model_stats)
        return overview


def _build_summary_data(data, target):
    df = data.groupby(target).agg(['mean', 'median', 'min', 'max', 'std', 'size', 'count'])
    df = df.swaplevel(0, 1, axis=1)
    df = df.unstack(level=1)
    df = df.swaplevel(0, 2)
    df = df.unstack(level=2)
    df = df.swaplevel(0, 1)
    df.sort_index(axis=0, level=0, inplace=True)
    df['na_percentage'] = 1 - (df['count'] / df['size'])
    return df


def plot_missing_values_per_group(data_summary, target):
    fig = go.Figure()
    for name, df in data_summary.groupby(target):
        fig.add_trace(go.Bar(
            x=df.index.levels[0],
            y=df['na_percentage'],
            name=name,
        ))
        fig.update_layout(
            title={
                'text': 'missing values per group',
                'font': {'size': 25}
            },
            template='plotly_white'
        )
    return fig


def read_in_files(filenames):
    """
    read-in file-names from a list of strings and returns a list of lies of same length
    :param filenames: list of filenames as string
    :return: list of files
    """
    files = []
    for idx, item in enumerate(filenames):
        df = pd.read_csv(f'{item}')
        df.rename(columns={df.columns[0]: "id"}, inplace=True)
        file = df.copy()
        files.append(file)
    return files


def get_first_value(df, feature):
    for item in feature:
        temp = [i for i in df.index.to_list() if item in i]
        a = df[str(temp[0])]
        b = df[str(temp[1])]
        name = f"{item}_"
        if pd.isnull(a) and not pd.isnull(b):
            df[name] = b
        else:
            df[name] = a
        df.drop(temp, inplace=True)
    return df


def replace_text_with_number(x):
    """
    takes numeric information at first place as cell content and drops string information
    :param x: mixed(numeric and string information) panda.series
    :return: numerical panda.series
    """
    for idx, item in enumerate(x):
        if pd.isna(item) is False:
            try:
                x_temp = float(item)
            except:
                x_temp = item.split()[0]
                if any(char.isdigit() for char in x_temp):  # checks if there are numbers in string
                    x_temp = float(x_temp.replace(',', '.'))
                else:
                    x_temp = np.nan  # if it's only text information set value to 'nan'
            x[idx] = x_temp
    return x


def flatten(li):
    temp_cols = []
    for item in li:
        if not isinstance(item, list):
            temp_cols.append([item])
        else:
            temp_cols.append(item)
    flat_list = [item for sublist in temp_cols for item in sublist]
    return flat_list

# FOR ModelEvaluation

#  working_file = "working_file_long_2021-03-22.pkl"
#     # simulation = "test"
#     size_test_data = 0.25
#     # chosen_model = 'SVC'
#
#     evaluation = ModelEvaluation(working_file=working_file,
#                                  target='final_result',
#                                  simulation_file=f'train_test_split_simulation.csv',
#                                  pipe=f'pipe.joblib')
#
#     # evaluation.plot_pipe(store_path=None, type='pipe')
#     # evaluation.chose_model(test_size=0.25, model='SVC', sorted_by=('test_accuracy', 'mean_cv_accuracy'))
#     # evaluation.plot_pipe(store_path=None, type='best_model')
#     # evaluation.plot_scoring_overview(store_path=None)
#
#     mod = evaluation.analyze_models(test_size=0.25, models=['LogisticRegression'], no_features=3)
#     pprint(mod)
#
#     # for algorithm in ['LogisticRegression', 'KNeighborsClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC']:
#     #     evaluation.chose_model(test_size=0.25, model=algorithm, sorted_by=('test_accuracy', 'mean_cv_accuracy'))
#     #     print(evaluation.get_algorithm())
#     #     print(evaluation.get_stats_model())
#     #     print(evaluation.get_best_param())
#     #     evaluation.calculate_feature_importance(no_features=10)
#     #     print(evaluation.get_most_important_features(type='list'))
#     #     print(evaluation.get_most_important_features(type='df'))
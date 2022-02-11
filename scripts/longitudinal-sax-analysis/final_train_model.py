import os
import itertools
import joblib
import pandas as pd
import shutil
from helpers import flatten
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from pathlib import Path

# own transformers
from sklearn_custom.feature_selection.VarianceThreshold import VarianceThreshold
from sklearn_custom.imputers.SimpleImputer import SimpleImputer
from sklearn_custom.encoders.OneHotEncoder import OneHotEncoder
from sklearn_custom.transformers.ColumnTransformer import ColumnTransformer
from sklearn_custom.preprocessing.MinMaxScaler import MinMaxScaler
from sklearn_custom.transformers.SAXTransformer import SAX_Transformer
from sklearn_custom.imputers.KNNImputer import KNNImputer

from config import working_file, target, random_states, test_sizes, scoring, \
    sax_transformer, sax_coded, num_coded, bin_coded, include_cols, params, store_path


def run_final_train_model():
    data = pd.read_pickle(Path(f"{store_path}/{working_file}.pkl"))

    X = data.drop([target], axis=1)
    y = data[target]

    # builds names of columns which should be taken into account (static)
    include_cols_fl = flatten(include_cols)
    sax_pipeline_names = [i for i in data.columns if i.startswith(sax_coded) and i.split('_')[0] in include_cols_fl]
    num_pipeline_names = [i for i in data.columns if i.startswith(
        num_coded) and i in include_cols_fl]  # num_coded  # collect 'temp_<36', 'temp_36_36.9', 'temp_37_38', 'temp_>38' (preprocessing)
    cat_pipeline_names = [i for i in data.columns if i.startswith(bin_coded) and i in include_cols_fl]  # bin_coded

    if os.path.exists(store_path) is False:
        try:
            os.mkdir(store_path)
        except FileNotFoundError:
            raise Exception(f"File not found!")

    # DEFINE: 'pipe'!
    pipe = Pipeline(steps=[('prep',
                            ColumnTransformer(transformers=[('num_pipeline',
                                                             Pipeline(steps=[
                                                                 ('imputer_num', KNNImputer()),
                                                                 ('min_max', MinMaxScaler()),
                                                                 ('zero_var',
                                                                  VarianceThreshold(threshold=(.95 * (1 - .95))))
                                                             ]),
                                                             num_pipeline_names
                                                             ),

                                                            ('sax_pipeline',
                                                             Pipeline(steps=[
                                                                 ('sax',
                                                                  SAX_Transformer(
                                                                      n_letters=sax_transformer['n_letters'],
                                                                      n_length=sax_transformer['n_length'],
                                                                      scaler=sax_transformer['scaler'],
                                                                      thresholds=sax_transformer[
                                                                          'thresholds'],
                                                                      cat_ordered=sax_transformer[
                                                                          'cat_ordered'])),
                                                                 ('imputer_sax', SimpleImputer(strategy='constant',
                                                                                               fill_value='MISSING',
                                                                                               df_out=True)),
                                                                 ('oh', OneHotEncoder(sparse=False, df_out=True,
                                                                                      new_cats=True,
                                                                                      fill_value='MISSING',
                                                                                      drop='first'))
                                                             ]),
                                                             sax_pipeline_names
                                                             ),

                                                            ('cat_pipeline',
                                                             Pipeline(steps=[
                                                                 ('imputer_cat', KNNImputer(df_out=True)),
                                                                 ('zero_var',
                                                                  VarianceThreshold(threshold=(.95 * (1 - .95)),
                                                                                    df_out=True)),
                                                             ]),
                                                             cat_pipeline_names
                                                             )

                                                            ],
                                              remainder='drop',
                                              n_jobs=-1)),
                           ('estimator', None),
                           ])

    joblib.dump(pipe, f'{store_path}/pipe.joblib')
    joblib.dump(params, f'{store_path}/params.joblib')

    simulations = list(itertools.product(random_states, test_sizes))
    data = []

    if os.path.exists(f'{store_path}/cv_results') is False:
        try:
            os.mkdir(f"{store_path}/cv_results")
        except FileNotFoundError:
            raise Exception(f"File not found!")

    for i, (random_state, test_size) in enumerate(simulations):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        y_train_bin = y_train.map({'survived': 1, 'died': 0})  # for precision_scorer labels have to be binary with 0,1
        y_test_bin = y_test.map({'survived': 1, 'died': 0})

        results = {}
        for idx, algorithm in enumerate(params[1]):
            algorithm_copy = algorithm.copy()
            if len(params[0]) != 0:
                updated_params = params[0][0].copy()
            else:
                updated_params = {}
            estimator = algorithm_copy.pop('estimator')[0]
            pipe.steps.pop(-1)  # drop estimator (last step in pipeline)
            pipe.steps.append(('estimator', estimator))  # add new estimator
            updated_params.update(algorithm_copy)

            kfold = KFold(n_splits=5, shuffle=False)  # , random_state=123)

            model = GridSearchCV(estimator=pipe, param_grid=updated_params, cv=kfold, n_jobs=-1,
                                 return_train_score=True,
                                 scoring=scoring, refit=scoring[0])
            model.fit(X_train, y_train_bin)
            e = str(estimator)[:-2]
            results.update({e: model})

        # collect results
        dfs = []
        for key, value in results.items():
            this_df = pd.DataFrame(value.cv_results_)
            this_df['name'] = key
            dfs.append(this_df)
        df = pd.concat(dfs, ignore_index=True)
        df = df.loc[:, ~ df.columns.str.contains("param_")]

        df.to_csv(f'{store_path}/cv_results/cv_results_simulation_{i}.csv')
        joblib.dump(results, f"{store_path}/cv_results/results_simulation_{i}.joblib")

        for key, value in results.items():
            y_pred = results.get(key).best_estimator_.predict(X_test)
            row = [key, i,
                   results.get(key).best_params_, results.get(key).best_score_,
                   results.get(key).cv_results_.get('std_test_accuracy')[results.get(key).best_index_],
                   results.get(key).cv_results_.get('mean_train_accuracy')[results.get(key).best_index_],
                   results.get(key).cv_results_.get('std_train_accuracy')[results.get(key).best_index_],
                   accuracy_score(y_test_bin, y_pred), test_size, random_state]
            data.append(row)

    data = pd.DataFrame(data)
    data.columns = ['algorithm', 'simulation', 'best_params', 'mean_cv_accuracy', 'cv_std', 'mean_train_accuracy',
                    'train_std', 'test_accuracy', 'test_size', 'random_state']
    data.to_csv(f'{store_path}/train_test_split_simulation.csv')

    # store this file and it's config in actual 'simulation_name'-folder
    file = f"{os.getcwd()}/final_train_model.py"
    config = f"{os.getcwd()}/config.py"
    shutil.copy(file, store_path)
    shutil.copy(config, store_path)


if __name__ == '__main__':
    run_final_train_model()


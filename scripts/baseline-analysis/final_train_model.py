import os
import itertools
import joblib
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.compose import make_column_selector
from config import target, random_states, test_sizes, scoring, params, \
    working_file, store_path


def run():
    data = pd.read_pickle(f"{store_path}/{working_file}_adjusted.pkl")

    X = data.drop([target], axis=1)
    y = data[target]

    pipe = Pipeline(steps=[('prep',
                            ColumnTransformer(transformers=[('num_pipeline',
                                                             Pipeline([
                                                                 ('imputer', KNNImputer()),
                                                                 ('log_trans', FunctionTransformer(np.log1p)),
                                                                 ('min_max', MinMaxScaler())
                                                             ]),
                                                             make_column_selector(dtype_include=np.number)
                                                             ),

                                                            ('cat_pipeline',
                                                             Pipeline([
                                                                 ('oe', OrdinalEncoder())
                                                             ]),
                                                             make_column_selector(dtype_exclude=np.number)

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
        print(f"simulation: {i}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        y_train_bin = y_train.map(
            {'positive': 1, 'negative': 0})  # for precision_scorer labels have to be binary with 0,1
        y_test_bin = y_test.map({'positive': 1, 'negative': 0})

        results = {}
        for idx, algorithm in enumerate(params[1]):
            print(f"simulation {i}.{idx}: {algorithm}")
            algorithm_copy = algorithm.copy()
            if len(params[0]) != 0:
                updated_params = params[0][0].copy()
            else:
                updated_params = {}
            estimator = algorithm_copy.pop('estimator')[0]
            pipe.steps.pop(-1)  # drop estimator (last step in pipeline)
            pipe.steps.append(('estimator', estimator))  # add new estimator
            updated_params.update(algorithm_copy)

            kfold = KFold(n_splits=5, shuffle=True, random_state=123)

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
    print(f"final_train_model finished!")


if __name__ == '__main__':
    run()

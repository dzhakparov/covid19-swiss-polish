from src.helpers import log
from config import general, sax_pipe, ml_params
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle


from sklearn_custom.transformers.ColumnTransformer import ColumnTransformer
from sklearn_custom.transformers.SAXTransformer import SAX_Transformer


@log()
def run(random_state):

    X, y = get_data()

    X_train, y_train, X_test, y_test = build_train_test(X, y, random_state)

    # shuffle y (only use for baseline estimation!)
    if ml_params['shuffle_target']:
        if y_train is not None:
            index = shuffle(y_train.index.to_list(), random_state=ml_params['shuffle_state'])
            values = shuffle(y_train.values, random_state=ml_params['shuffle_state'])
            y_train = pd.Series(data=values, index=index)

        if y_test is not None:
            index = shuffle(y_test.index.to_list(), random_state=ml_params['shuffle_state'])
            values = shuffle(y_test.values, random_state=ml_params['shuffle_state'])
            y_test = pd.Series(data=values, index=index)

    X_train, X_test = sax(X_train, X_test)  # sax-transformation of defined features

    _store_calculated_data(X_train, y_train, X_test, y_test)

    return X_train, X_test, y_train, y_test


@log()
def _store_calculated_data(X_train, y_train, X_test, y_test):

    X_train.to_csv(f"{general['output_path']}X_train_after_sax.csv", index=False)
    X_train.to_pickle(f"{general['output_path']}X_train.pkl")
    y_train.to_pickle(f"{general['output_path']}y_train.pkl")

    if X_test is not None:
        cols = X_train.columns.to_list()  # same order of columns
        X_test = X_test[cols]
        X_test.to_csv(f"{general['output_path']}X_test_after_sax.csv", index=False)
        X_test.to_pickle(f"{general['output_path']}X_test.pkl")
        y_test.to_pickle(f"{general['output_path']}y_test.pkl")


@log()
def get_data():
    try:
        df = pd.read_pickle(f"{general['output_path']}{general['output_filename']}.pkl")
    except FileNotFoundError:
        pass

    df.drop(ml_params['remove_cols'], axis=1, inplace=True)

    X = df.drop(ml_params['target'], axis=1)
    y = df[ml_params['target']]
    y = pd.DataFrame(y, columns=[ml_params['target']])

    return X, y


@log()
def build_train_test(X, y, random_state):

    if ml_params['test_size'] != 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ml_params['test_size'],
                                                            random_state=random_state)
        y_train = y_train.iloc[:,0]
        y_test = y_test.iloc[:, 0]
    else:
        X_train = X
        X_test = None
        y_train = y.iloc[:,0]
        y_test = None

    return X_train, y_train, X_test, y_test


@log()
def sax(X_train, X_test):

    sax_names = []
    for item in X_train.columns.to_list():
        for i in sax_pipe['sax_groups']:
            if i in item:
                sax_names.append(item)

    sax_pipeline = Pipeline([
        ('sax', SAX_Transformer(n_letters=sax_pipe['sax_transformer']['n_letters'],
                                n_length=sax_pipe['sax_transformer']['n_length'],
                                scaler=sax_pipe['sax_transformer']['scaler'],
                                thresholds=sax_pipe['sax_transformer']['thresholds'],
                                cat_ordered=sax_pipe['sax_transformer']['cat_ordered']))
    ])

    sax_transformation = ColumnTransformer([
        ('saxpipe', sax_pipeline, sax_names)
    ],
        remainder=sax_pipe['sax_transformer']['reminder']
    )

    # apply
    X_train = sax_transformation.fit_transform(X_train)
    borders = sax_transformation.named_transformers_.saxpipe.named_steps.sax.get_borders()

    with open(f"{general['output_path']}/borders_sax.csv", 'w') as f:
        for key in borders.keys():
            f.write("%s,%s\n" % (key, borders[key]))

    if X_test is not None:
        X_test = sax_transformation.transform(X_test)

    transformed_columns = [i[:-1] for i in sax_pipe['sax_groups']]

    return X_train, X_test


if __name__ == '__main__':
    pass

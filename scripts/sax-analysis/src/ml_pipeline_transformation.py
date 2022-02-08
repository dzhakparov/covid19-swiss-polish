from config import general, ml_params, sax_params
from config import preprocessing as prep

from sklearn_custom.transformers.ColumnTransformer import ColumnTransformer
import numpy as np
from src.plots import fig_heatmap
from src.helpers import store_figure, log, get_train_test_files
from config import num_pipeline, cat_bin_pipeline, cat_nom_pipeline, cat_ord_pipeline


@log()
def run():
    """
    transforms X-part of train- and test-data to machine learning data (scaled, with no nan values, dropped features, ...)
    and stores it as X_train_ml.pkl and X_test.pkl
    """

    # script = str.split(os.path.basename(__file__), '.')[0]
    # logging, console = setup(script)
    # logging.info(f"parameters from 'ml_params' ('config.py'):\n\n {pformat(ml_params)}\n")

    X_train, X_test, y_train, y_test = get_train_test_files(['X_train', 'X_test', 'y_train', 'y_test']
                                                            # logging, error_message=f"file not found! Run "
                                                            #                        f"'ml_pipeline_sax' first to create "
                                                            #                        f"train- and test-files"
                                                            )

    # dropping features according to parameters in 'remain_cols', 'remove_cols' and 'drop_frac_na_cols' in ml_params
    X_train, X_test = drop_features(X_train, X_test)

    X_train_ml, X_test_ml = calculate_pipeline(X_train, X_test)

    # drop correlated features
    if 1 > ml_params['corr_threshold'] >= 0:
        X_train_ml, X_test_ml = drop_correlated_features(X_train_ml, X_test_ml)
    else:
        pass
        # logging.info("")

    _store_calculated_data(X_train_ml, X_test_ml, X_train, X_test)


@log()
def drop_correlated_features(X_train_ml, X_test_ml):
    # dropping features due high correlation
    corr_matrix = X_train_ml.corr(method=ml_params['corr_method'])

    # filtering correlations
    if ml_params['corr_filter'] is not None and len(corr_matrix) > 0:
        corr_matrix = corr_matrix.apply(
            lambda x: np.where(abs(x).between(ml_params['corr_filter'][0], ml_params['corr_filter'][1]),
                               x, np.nan))

        # Draw the heatmap with plotly
        fig = fig_heatmap(corr_matrix, ml_params)
        store_figure(fig, name='correlation_matrix', path=f"{general['output_path']}fig/",
                     format=general['image_format'],
                     show_browser=general['show_browser'], width=1200, height=1200)
        # logging.info(f"successfully plotted: {general['output_path']}fig/correlation_matrix.svg\n")

        # Select upper triangle of matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than this threshold
        to_drop = [column for column in upper.columns if any(abs(upper[column]) > ml_params['corr_threshold'])]

        corr_dict = {}
        # find correlated items:
        for item in to_drop:
            high_correlated = corr_matrix[item][abs(corr_matrix[item]) > ml_params['corr_threshold']]
            dict_high_correlated = high_correlated[high_correlated.index != item].to_dict()
            corr_dict.update({item: dict_high_correlated})

        # logging.info(f"dropped {len(to_drop)} features due high correlation > {ml_params['corr_threshold']}:"
        #              f"\n\n{pformat(to_drop)}\n")

        if corr_dict:
            pass
            # logging.info(f"dropped features are correlated with the following features:\n {pformat(corr_dict)}\n")

        # Drop features
        if len(to_drop) > 0:
            X_train_ml = X_train_ml.drop(to_drop, axis=1)

            if X_test_ml is not None:
                X_test_ml = X_test_ml.drop(to_drop, axis=1)
    else:
        if len(corr_matrix) == 0:
            pass
            # logging.info(f'dropped 0 features due high correlation: {ml_params["corr_threshold"]}\n')

    return X_train_ml, X_test_ml


@log()
def _store_calculated_data(X_train_ml, X_test_ml, X_train, X_test):
    X_train_ml.to_csv(f"{general['output_path']}X_train_after_ml_transformation.csv", index=False)
    X_train.to_csv(f"{general['output_path']}X_train_before_ml_transformation.csv", index=False)
    X_train_ml.to_pickle(f"{general['output_path']}X_train_ml.pkl")

    if X_test_ml is not None:
        X_test_ml.to_csv(f"{general['output_path']}X_test_after_ml_transformation.csv", index=False)
        X_test.to_csv(f"{general['output_path']}X_test_before_ml_transformation.csv", index=False)
        X_test_ml.to_pickle(f"{general['output_path']}X_test_ml.pkl")


@log()
def drop_features(X_train, X_test):
    if len(ml_params['remain_cols']) != 0 and len(ml_params['remove_cols']) != 0:
        # logging.warning(f"'remain_cols' and 'remove_cols' can't be both filled... NOTHING DONE!\n"
        #                 f"remain_cols: {ml_params['remain_cols']}\n"
        #                 f"remove_cols: {ml_params['remove_cols']}\n")
        return X_train, X_test

    if len(ml_params['remain_cols']) != 0 and len(ml_params['remove_cols']) == 0:
        remain = [i for i in ml_params['remain_cols'] if i in X_train.columns]
    elif len(ml_params['remain_cols']) == 0 and len(ml_params['remove_cols']) != 0:
        remain = [i for i in X_train.columns.to_list() if i not in ml_params['remove_cols']]
    elif len(ml_params['remain_cols']) == 0 and len(ml_params['remove_cols']) == 0:
        remain = X_train.columns.to_list()

    drop = [i for i in X_train.columns.to_list() if i not in remain]

    if len(remain) != 0:
        X_train = X_train[remain]
        X_test = X_test[remain]
        if len(drop) > len(remain):
            pass
            # logging.info(f"remaining features for machine learning ({len(X_train.columns)}): \n{pformat(remain)}\n")
        else:
            pass
            # logging.info(f"dropped {len(drop)} features for machine learning: \n{pformat(drop)}\n")
    else:
        pass
        # logging.info(f"no remaining features found! Therefore X_train and X_test will be untouched!\n")

    # remove columns which exceed threshold defined in ml_params['drop_frac_na_cols']
    drop_due_threshold = X_train.isnull().mean() > ml_params['drop_frac_na_cols']
    X_train = X_train.loc[:, -drop_due_threshold]
    X_test = X_test.loc[:, X_train.columns]
    # logging.info(f"dropping {len(drop_due_threshold[drop_due_threshold == True])} "
    #              f"columns due exceeding threshold of missing values: "
    #              f"{ml_params['drop_frac_na_cols']}:\n"
    #              f"{pformat(list(drop_due_threshold[drop_due_threshold == True].index))}\n")

    # logging.info(f"remaining features for machine learning ({len(X_train.columns)}): \n"
    #              f"{pformat(X_train.columns.to_list())}\n")

    return X_train, X_test


@log()
def calculate_pipeline(X_train, X_test):
    # numerical features
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_num = X_train.select_dtypes(include=numerics)
    num_names = df_num.columns.to_list()

    # categorical features
    df_cat = X_train.select_dtypes(include='category')
    cat_ord_names = [i for i in df_cat.columns if
                     df_cat[i].cat.ordered and i not in prep['convert_categorical_binaries']]
    cat_nom_names = [i for i in df_cat.columns if
                     not df_cat[i].cat.ordered and i not in prep['convert_categorical_binaries']]
    cat_bin_names = [i for i in df_cat.columns if i in prep['convert_categorical_binaries']]

    preprocessing = ColumnTransformer([
        ('num', num_pipeline, num_names),
        ('cat_ord', cat_ord_pipeline, cat_ord_names),
        ('cat_oh', cat_nom_pipeline, cat_nom_names),
        ('cat_bin', cat_bin_pipeline, cat_bin_names)
    ])

    X_train_ml = preprocessing.fit_transform(X_train)

    # logging pipelines
    # logging.info(f"\n\nnum_pipeline: \n\n{num_pipeline}, \n\napplied on\n {pformat(list(num_names))}\n")
    # logging.info(f"\n\ncat_ord_pipeline: \n\n{cat_ord_pipeline}, \n\napplied on\n {pformat(list(cat_ord_names))}\n")
    # logging.info(f"\n\ncat_nom_pipeline: \n\n{cat_nom_pipeline}, \n\napplied on\n {pformat(list(cat_nom_names))}\n")
    # logging.info(f"all 3 pipelines are passed to ColumnTransformer\n")
    # logging.info(f"new shape X_train: {X_train.shape}\n")

    # logging.info(f"all columns in X: \n\n{pformat(sorted(X_train.columns.to_list()))}\n")
    #
    # logging.info(f"X_train_ml: \n{X_train_ml.head(5)}\n")

    if X_test is not None:
        X_test_ml = preprocessing.transform(X_test)
        # logging.info(f"X_test_ml: \n{X_test_ml.head(5)}\n")
    else:
        X_test_ml = None

    # remove only '_MISSING' column after one hot encoding  (information about time in hospital,
    # not only about medical feature)
    if ml_params['dropping_MISSING']:
        dropping_cols = [i for i in X_train_ml.columns if
                         i.endswith('_MISSING') and i.split('_')[0] + '_' in sax_params['features']]
        X_train_ml.drop(columns=dropping_cols, inplace=True)
        if X_test is not None:
            X_test_ml.drop(columns=dropping_cols, inplace=True)
        # logging.info(f"dropping sax-features-columns '_MISSING' ({len(dropping_cols)}): {pformat(dropping_cols)}\n")
        # logging.info(f"remaining columns in X_train_ml and X_test_ml ({len(X_train_ml.columns)}):"
        #              f" \n {pformat(X_train_ml.columns.to_list())}\n")

    return X_train_ml, X_test_ml


if __name__ == '__main__':
    pass

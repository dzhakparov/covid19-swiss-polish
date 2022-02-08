import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from src.helpers import get_train_test_files, log
from joblib import dump, load
from config import general, feature_selection

# Step forward feature selection
# Step forward feature selection starts by training a machine learning model for each feature in the dataset
# and selects, as the starting feature, the one that returns the best performing model, according to a certain
# evaluation criteria we choose.
# In the second step, it creates machine learning models for all combinations of the feature selected in the
# previous step and a second feature. It selects the pair that produces the best performing algorithm.

# Step backward feature selection
# Step Backward Feature Selection starts by fitting a model using all features in the data set and determining its
# performance.
# Then, it trains models on all possible combinations of all features -1, and removes the feature that returns the
# model with the lowest performance.
# In the third step it trains models in all possible combinations of the features remaining from step 2 -1 feature,
# and removes the feature that produced the lowest performing model.


@log()
def run():

    # load in fitted models
    fitted_models = load(f"{general['output_path']}fitted_models.joblib")

    # load in train and test data
    X_train_ml, X_test_ml, y_train, y_test = get_train_test_files(
        ['X_train_ml', 'X_test_ml', 'y_train', 'y_test'],
        error_message=f"file not found! Run 'ml_pipeline_sax' first to create train- and test-files and "
                      f"'ml_pipeline_transformation' to generate 'X_train_ml' and 'X_test_ml'")

    results = fw_bw_feature_selection(fitted_models, X_train_ml, y_train,
                                      **feature_selection['fw_bw_feature_selection'])

    dump(results, f"{general['output_path']}feature_selection.joblib")  # store results to use for evaluation


@log()
def fw_bw_feature_selection(fitted_models, X_train_ml, y_train, **kwargs):

    algorithms = list(fitted_models.keys())

    if 'k_features' not in kwargs:
        k_features = len(X_train_ml.columns)
    else:
        k_features = kwargs.pop('k_features')
        if k_features > len(X_train_ml.columns):
            k_features = len(X_train_ml.columns)

    results = {}

    for algorithm in algorithms:
        if algorithm != 'VotingClassifier':
            best_estimator = fitted_models[algorithm].best_estimator_.named_steps.classifier

            # http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
            sfs = SFS(estimator=best_estimator, k_features=k_features, **kwargs)

            sfs = sfs.fit(np.array(X_train_ml), y_train, custom_feature_names=list(X_train_ml.columns))
            selected_feat = X_train_ml.columns[list(sfs.k_feature_idx_)]

            results[algorithm] = sfs

    return results


if __name__ == '__main__':
    pass

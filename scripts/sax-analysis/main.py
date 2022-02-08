from datetime import datetime
from src.setup import setup_system

from src.file_preparation import read_in_files
from config import general, eda, preprocessing, ml_params, sax_params
from src.helpers import store_results, get_console
from src import data_preprocessing, data_EDA, data_SAX, ml_pipeline_sax, ml_pipeline_transformation, \
    ml_pipeline_predictions, ml_pipeline_evaluations, ml_pipeline_feature_selection, feature_selection_evaluation, \
    evaluate_simulation

# from src.evaluate_simulation import evaluate_simulation
from src.plots import boxplot_sax_features, survivalplot_sax_features, visualize_matrix, matrix_plots
# from joblib import dump, load

# TODO: methodology
# TODO: comments in code
# TODO: remove unused code (exp. packages, import and code)
# TODO: build 'store' folder in setup system
# TODO: extract possibility to use only evaluate simulation
# TODO: synchronize methodology with document on overleaf
# TODO: add config_classifier to stored files

# "Warnings"
# TODO: /home/schmidmarco/Documents_not_synchronized/CODE/PROJECTS/covid19_publication/src/data_preprocessing.py:357: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead.  To get a de-fragmented frame, use `newframe = frame.copy()`
#   file[name] = file.loc[:,nom[idx]] / file.loc[:,denom[idx]]
# TODO: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.

if __name__ == '__main__':

    setup_system()  # OKAY
    console = get_console()  # OKAY
    timestamp_sim = str(datetime.now())  # OKAY

    files = read_in_files(filenames=general['data_files'])  # OKAY
    renaming_files = read_in_files(filenames=preprocessing['renaming_files'])  # OKAY
    file = data_preprocessing.run(files=files, renaming_files=renaming_files)  # OKAY

    # simulation: list of random states in ml_Params: 'random_state': [111, 123, ...]
    simulations = ml_params['random_state']
    for idx, sim in enumerate(simulations):

        console.info(f"\nSIMULATION {idx + 1} of {len(simulations)} ({round(((idx + 1) / len(simulations)) * 100, 1)})%\n")

        # 'static' analysis only maximal one time per simulation (does not change with train-, test-split)
        if idx == 0:

            file_mod = file.drop(["date_hospitalisation"], axis=1)

            if 'visualize_matrix' in general['visualizations']:
                fig = visualize_matrix(file_mod, sort_by=["age"])
                file_mod = file.drop(['date_hospitalisation'], axis=1)

            if 'matrix_plot' in general['visualizations']:
                matrix_plots(file_mod)

            if eda['static_analysis']:
                data_EDA.run()

            if sax_params['static_analysis']:
                data_SAX.run()

        # sax-transformation: ml_pipeline_sax
        X_train, X_test, y_train, y_test = ml_pipeline_sax.run(random_state=sim)

        # optional 'dynamical' analysis (does change with different train-, test-split
        # only the last simulation is stored...
        if 'boxplot_sax_features' in general['visualizations']:
            boxplot_sax_features(X_train, y_train, 'max_mews')
        if 'survivalplot_sax_features' in general['visualizations']:
            survivalplot_sax_features(X_train, y_train)

        # machine-learning algorithm: ml_pipeline_transformation
        ml_pipeline_transformation.run()

        # apply machine learning to train and test data (models and parameters from config)
        ml_pipeline_predictions.run()

        # run evaluations (plots and metrics) on stored models
        ml_pipeline_evaluations.run()

        # feature selection
        ml_pipeline_feature_selection.run()  # takes some time...
        feature_selection_evaluation.run()

        # store results and logs
        timestamp = str(datetime.now())
        store_results(timestamp_sim=timestamp_sim, timestamp=timestamp)
        # store_results_cloud(timestamp_sim=timestamp_sim, timestamp=timestamp)
        setup_system()
        # clear_output()  # drops folder 'output'

    # evaluate simulation
    # timestamp_sim = input("")
    # timestamp_sim = '2022-02-07 18:58:41.540391'
    evaluate_simulation.evaluate_simulation(timestamp_sim=timestamp_sim)
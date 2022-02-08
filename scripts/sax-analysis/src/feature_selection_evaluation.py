from src.helpers import get_train_test_files
from joblib import load
from config import general, feature_selection
from src.helpers import store_figure, log
import plotly.graph_objects as go
import numpy as np
import math
import pandas as pd
from functools import reduce


@log()
def run():

    # script = str.split(os.path.basename(__file__), '.')[0]
    # logging, console = setup(script)

    fs = load(
        f"{general['output_path']}feature_selection.joblib")  # dictionary with key='Algorithm-name" and value=SFS-object

    # X_train_ml, X_test_ml, y_train, y_test = get_train_test_files(
    #     ['X_train_ml', 'X_test_ml', 'y_train', 'y_test'], logging=None)

    fitted_models = load(f"{general['output_path']}fitted_models.joblib")

    fig = plot_feature_selection(fs, error_visability=False)  # attention: only for 'forward selection' at the moment!
    store_figure(fig, name='figure_selection', path=f"{general['output_path']}fig/", format=general['image_format'],
                 show_browser=general['show_browser'], width=1200, height=800)
    # logging.info(f"plot 'feature_selection' successfully build in folder {general['output_path']}fig/")

    build_excel_results(fs)
    # logging.info(f"build excel-sheet with feature_selection of each algorithm successfully"
    #              f" in folder {general['output_path']}")

    df = build_overall_ranking(fs)
    df.to_csv(f"{general['output_path']}feature_overall_ranking.csv")
    # logging.info(f"build successfully feature_overall_ranking.csv in folder {general['output_path']}")

    fig = plot_overall_ranking(fs)
    store_figure(fig, name='feature_overall_score', path=f"{general['output_path']}fig/", format=general['image_format'],
                 show_browser=general['show_browser'], width=1200, height=800)
    # logging.info(f"plot 'feature_overall_score' build in folder {general['output_path']}fig/")

    # build_csv(fs, 'AdaBoost')


@log()
def plot_overall_ranking(fs, algorithms=None):

    if algorithms is None:
        alg = [i for i in fs.keys()]
    else:
        alg = algorithms

    table = build_overall_ranking(fs, alg)

    fig = go.Figure([go.Bar(x=table.index, y=table['score'])])

    fig.update_layout(
        title={
            'text': f"overall feature importance: algorithms = {alg}",
            'font': {'size': 25}
        },

        yaxis_title=f"average score",
        xaxis_title="features",
        template='plotly_white'
    )

    return fig


@log()
def build_overall_ranking(fs, algorithms=None):
    """
    calculates overall importance of each feature
    :param fs: SFS object
    :param algorithms: list of desired algorithm names (None (default) = all available algorithms)
    :return: rankig DataFrame
    """
    if algorithms is None:
        alg = [i for i in fs.keys()]
    else:
        alg = algorithms

    ranking = []

    for algorithm in fs:
        if algorithm not in alg:
            continue
        else:
            data = fs[algorithm]
            feature_order = []
            for idx in range(2, len(data.subsets_)+1):
                f1 = list(data.subsets_[idx - 1]['feature_names'])
                f2 = list(data.subsets_[idx]['feature_names'])
                f_new = [x for x in f2 if x not in f1]
                feature_order.append(f_new[0])
            feature_order.insert(0, data.subsets_[1]['feature_names'][0])
            ranking.append(pd.DataFrame(index=feature_order, data=list(range(len(feature_order), 0, -1)), columns=[algorithm]))

    df = reduce(lambda left, right: left.join(right, how='outer'), ranking)  # join DataFrames of all algorithms
    overall_sum = df.sum(axis=1).sum(axis=0)
    df['sum'] = df.sum(axis=1)
    df['score']  = df.loc[:,"sum"] / overall_sum
    df.sort_values('score', ascending=False, inplace=True)
    return df


@log()
def build_excel_results(fs):
    with pd.ExcelWriter(f"{general['output_path']}results_feature_selection.xlsx") as writer:
        for algorithm in fs:
            res = pd.DataFrame.from_dict(fs[algorithm].get_metric_dict()).T
            res.to_excel(writer, sheet_name=algorithm)


@log()
def build_csv(fs, name):
    try:
        res = pd.DataFrame.from_dict(fs[name].get_metric_dict()).T
    except:
        print(f"object with this name does not exist!")

    res.to_csv(f"{general['output_path']}feature_selection_{name}.csv")


@log()
def plot_feature_selection(fs, error_visability=True):

    fig = go.Figure()

    for algorithm in fs:

        data = fs[algorithm]
        values = [data.subsets_[i]['avg_score'] for i in data.subsets_]
        errors = [np.std(data.subsets_[i]['cv_scores']) / math.sqrt(len(data.subsets_[i]['cv_scores'])) for
                  i in data.subsets_]  # standard error
        features = [list(data.subsets_[i]['feature_names']) for i in data.subsets_]

        # build list with new feature per step (with delta)
        new_feature = []
        new_delta = []
        for idx in range(2, len(data.subsets_) + 1):
            f1 = list(data.subsets_[idx - 1]['feature_names'])
            f2 = list(data.subsets_[idx]['feature_names'])
            f_new = [x for x in f2 if x not in f1]
            new_feature.append(f_new[0])
            new_delta.append(round(data.subsets_[idx]['avg_score'] - data.subsets_[idx - 1]['avg_score'], 3))

        new_feature.insert(0, data.subsets_[1]['feature_names'][0])
        new_delta.insert(0, '-')

        text_features = [f"<b>all:  </b> {features[i]} <br><b>new:</b> <i>{new_feature[i]}</i> (delta={new_delta[i]})"
                         for i in range(0, len(features))]

        # Add traces
        fig.add_trace(go.Scatter(x=list(range(1, len(values) + 1)),
                                 y=values,
                                 error_y=dict(
                                     type='data',  # value of error bar given in data coordinates
                                     array=errors,
                                     visible=error_visability),
                                 mode='markers+lines',
                                 name=algorithm,

                                 hovertemplate=
                                 '<br>no features: %{x}<br>' +
                                 'score: %{y} <br>' +
                                 '%{text}',
                                 text=text_features
                                 ))

    if error_visability:
        title = f"<b>Feature Selection (cv-mean and standard error, cv={data.cv})</b>"
    else:
        title = f"<b>Feature Selection (cv-mean)</b>"

    fig.update_layout(
        title={
            'text': title,
            'font': {'size': 25}
        },

        yaxis_title=f"{data.scoring}",
        xaxis_title="features",
        template='plotly_white'
    )

    return fig


if __name__ == '__main__':
    pass

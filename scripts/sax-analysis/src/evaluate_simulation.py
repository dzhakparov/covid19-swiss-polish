import pandas as pd
import os
from config import general
from src.helpers import get_wd
import plotly.graph_objects as go
import numpy as np
from src.helpers import store_figure
from src.ml_pipeline_evaluations import _build_positions
from plotly.subplots import make_subplots


def evaluate_simulation(timestamp_sim):

    # path = os.path.join(get_wd(), general['store_infos']['location'], timestamp)
    path_sim = os.path.join(get_wd(), general['store_infos']['location'], 'simulation_' + timestamp_sim)

    if not os.path.isdir(path_sim):
        raise Exception("folder with this timestamp does not exist!")

    sub_dirs = [name for name in os.listdir(path_sim) if os.path.isdir(os.path.join(path_sim, name)) and
                       not name.endswith('_cloud')]

    if len(sub_dirs) == 1:
        print(f"only one simulation in folder '{path_sim}', no evaluation is available!")
        exit()

    x = [[] for i in range(len(sub_dirs))]

    for idx, item in enumerate(sub_dirs):
        p = os.path.join(path_sim, item)
        for file in ['feature_overall_ranking', 'df_results']:
            data = pd.read_csv(f"{p}/{file}.csv", index_col=0)
            data['ts'] = item
            data['idx'] = idx
            x[idx].append(data)

    def Extract(lst, pos):
        return [item[pos] for item in lst]

    feature_importance = pd.concat(Extract(x, 0))
    train_test_scores = pd.concat(Extract(x, 1))

    alg = feature_importance.loc[:, :'sum'].columns.to_list()[:-1]

    # feature importance
    feature_importance.to_csv(path_or_buf=os.path.join(path_sim,"feature_importance_detail.csv"),index=True)
    feature_importance = feature_importance.reset_index().drop(['sum', 'idx'], axis=1)
    feature_importance = feature_importance.groupby(by='index').agg(['mean', 'std'])
    feature_importance.sort_values(('score','mean'), ascending=False, inplace=True)
    feature_importance.to_csv(path_or_buf=os.path.join(path_sim,"feature_importance.csv"),index=True)

    n = len(sub_dirs)
    fig = plot_feature_importance(feature_importance, n, alg)
    store_figure(fig, name='feature_importance', path=f"{path_sim}/", format=general['image_format'],
                 show_browser=general['show_browser'], width=1200, height=800)

    # train-test-scores
    train_test_scores.to_csv(path_or_buf=os.path.join(path_sim, "train_test_scores_detail.csv"), index=True)
    train_test_scores = train_test_scores.reset_index().drop(['ts', 'idx'], axis=1)
    train_test_scores = train_test_scores.groupby(by=['name', 'split']).agg(['mean', 'std'])
    train_test_scores.to_csv(path_or_buf=os.path.join(path_sim, "train_test_scores.csv"), index=True)

    fig = plot_train_test_scores(train_test_scores, n, sort='algorithm')
    store_figure(fig, name='train_test_scores_algorithm', path=f"{path_sim}/", format=general['image_format'],
                 show_browser=general['show_browser'], width=1200, height=800)

    fig = plot_train_test_scores(train_test_scores, n, sort='metric')
    store_figure(fig, name='train_test_scores_metric', path=f"{path_sim}/", format=general['image_format'],
                 show_browser=general['show_browser'], width=1200, height=800)


def _build_axis(train_test_scores):
    alg = train_test_scores.index.levels[0].to_list()
    scorer = train_test_scores.columns.levels[0].to_list()
    return [alg, scorer]


def _get_data(train_test_scores, l1, l2, x, sort='algorithm'):
    if sort == 'algorithm':
        data = train_test_scores.xs(x, axis=1, level=1, drop_level=False)
        data = list(data.loc[(l1, l2)])
    else:  # sort = 'columns'
        data = train_test_scores.xs(l2, axis=0, level=1, drop_level=False)
        data = list(data.loc[:,(l1, x)])
    return data


def plot_train_test_scores(train_test_scores, n, sort='algorithm', **kwargs):

    # sort = 'algorithm'
    # sort = 'metric'

    # train_test_scores (multilevel dataframe):
    #                     accuracy_score            ...  f1_score
    #                               mean       std  ...      mean       std
    # name          split                           ...
    # GNB           test        0.758170  0.071955  ...  0.670217  0.108623
    #               train       0.796667  0.035963  ...  0.710133  0.061648
    # KNN           test        0.732026  0.070878  ...  0.577531  0.103617
    #               train       0.813333  0.040661  ...  0.690275  0.083335
    # Random Forest test        0.764706  0.092801  ...  0.617770  0.161349
    #               train       0.837778  0.025531  ...  0.731629  0.040589

    # set train and test colors
    colors = []
    for item in train_test_scores.index.levels[1].to_list():
        if item.lower() in general['COLORS'].keys():
            colors.append(general['COLORS'][item])

    axis = _build_axis(train_test_scores)

    if sort == 'metric':
        axis[0], axis[1] = axis[1], axis[0]  # swap axis

    no_plots = len(axis[0])
    positions, geometry = _build_positions(no_plots)  # build geometry for desired number of figures

    fig = make_subplots(rows=geometry[0],
                        cols=geometry[1],
                        start_cell="top-left",
                        subplot_titles=axis[0])

    # define y-range of plot
    if 'y_range' not in kwargs:
        y_range = [0, 1]  # default
    else:
        if (isinstance(kwargs['y_range'], list) and len(
                kwargs['y_range']) == 2 and 0 <= kwargs['y_range'][0] < kwargs['y_range'][1] <= 1):
            y_range = kwargs['y_range']
        else:
            y_range = [0, 1]

    for index, alg in enumerate(axis[0]):
        x = axis[1]  # list of strings
        for idx, src in enumerate(train_test_scores.index.levels[1].to_list()):

            y = _get_data(train_test_scores, alg, src, 'mean', sort)  # list of numbers
            error_y = _get_data(train_test_scores, alg, src, 'std', sort)  # list of numbers

            fig.add_trace(go.Bar(name=f"{src} {alg}", x=x, y=y, error_y=dict(type='data', array=error_y),
                                 marker_color=colors[idx]),
                          row=positions[index][0], col=positions[index][1])
            fig['layout'][f'yaxis{idx + 1}'].update(range=y_range, autorange=False)

    fig.update_layout(
        title={
            'text': f"<b>Predictions with different Classifiers (sort = {sort}) <br> n={n}</b>",
            'font': {'size': 25}
        },
        template='plotly_white'
    )
    return fig


def plot_feature_importance(feature_importance, n, alg):
    fig = go.Figure([go.Bar(x=feature_importance.index,
                            y=feature_importance[('score', 'mean')],
                            error_y=dict(type='data', array=feature_importance[('score', 'std')] / np.sqrt(n),
                                         visible=True))])

    fig.update_layout(
        title={
            'text': f"overall feature importance with standard error <br> algorithms = {alg}",
             'font': {'size': 25}
        },

        yaxis_title=f"average score",
        xaxis_title="features",
        template='plotly_white'
    )

    return fig


if __name__ == '__main__':
    pass
    # timestamp_sim = '2022-02-07 18:58:41.540391'
    # evaluate_simulation(timestamp_sim=timestamp_sim)


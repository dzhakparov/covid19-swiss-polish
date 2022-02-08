import os
import numpy as np
import pandas as pd
import logging
from pprint import pformat
from datetime import datetime
from functools import reduce
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from statsmodels.distributions.empirical_distribution import ECDF
from config import general
from config import preprocessing as prep
import shutil
from shutil import copyfile

from functools import wraps
import logging
from pathlib import Path
import os.path
from datetime import datetime


def log(mode=None):
    """
    Args:
        mode: dictionary with settings exp.: mode = {'time': True}

    Returns:
    """

    params = {'time': True, 'doc': False, 'details': False}

    if mode is not None:
        params.update(mode)

    def log_int(fn):  # decorator

        path = os.path.join(get_wd(), general['output_path'], 'log')
        file = f"{path}/{fn.__module__}.log"

        @wraps(fn)
        def logging_fun(*args, **kwargs):

            with open(file, 'a+') as f:
                args_values_types = [(a, type(a)) for a in args]
                kwargs_values_types = [(k, v, type(v)) for k, v in kwargs.items()]

                if params['time']:
                    now = datetime.now()
                    f.write(f"\n{now}: Function '{fn.__name__}()' was called\n")
                else:
                    f.write(f"\nFunction '{fn.__name__}()' was called\n")

                if params['doc']:
                    f.write(f"{fn.__doc__}")

                # run function
                fn_results = fn(*args, **kwargs)

                if params['details']:
                    f.write(f"\nargs '{fn.__name__}': {pformat(args_values_types)}\n")
                    f.write(f"\nkwargs '{fn.__name__}': {pformat(kwargs_values_types)}\n")
                    f.write(f"\nFunction '{fn.__name__}' returns \n{pformat(fn_results)}\n")

                f.write(f"\n\tFunction '{fn.__name__}' was finished successfully\n")

                return fn_results

        return logging_fun

    return log_int


def get_console():
    """ get al logger instance for console
    :return: logger instance for console
    """
    level = logging.INFO
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s: %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setFormatter(formatter)
    console = logging.getLogger(f"console")
    console.setLevel(level)
    console.addHandler(handler)
    return console


def initialize_logger(file, path=None):
    if path is None:
        path = f"{general['output_path']}log/"
    else:
        path = f"{path}/"

    if os.path.exists(path) is False:
        try:
            os.mkdir(path)
        except OSError:
            print(f"Creation of the directory '{path}' failed")

    level = logging.INFO
    name = f"{path}{file}"
    log_file = f"{path}{file}.log"

    # FileHandler to log into file
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s: %(message)s')
    handler = logging.FileHandler(log_file)  # log into file
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # StreamHandler to log on console
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setFormatter(formatter)
    console = logging.getLogger(f"console_{file}")
    console.setLevel(level)
    console.addHandler(handler)

    logger.info(f"\n\n########## Covid19: {file}.py ##########\n".upper())
    console.info(f"RUNNING {file}")
    logger.info("setup successfully initialized\n")

    return logger, console


def get_wd():
    return os.getcwd()


def replace_cell_content(series, header, pattern, value):
    if ~series.name.find(header):
        if series.dtype == 'object':
            series = series.str.replace(pattern, value, regex=False)
            series = pd.to_numeric(series, errors='coerce')
    return series


def replace_cell_content_strings(x, new_value):
    for idx, item in enumerate(x):
        if pd.isna(item) is False:
            try:
                x_temp = float(item)
            except:
                x_temp = new_value
            x[idx] = float(x_temp)
    return x


def get_color_transparency(color, transparency):
    # replace rgb with rgba
    color = color.replace("rgb", "rgba")
    # replace ) with 'transparency and )
    color = color.replace(")", f", {transparency})")
    return color


def store_figure(fig, name=None, path=None, format='html', show_browser=False, **kwargs):
    if path is None:
        path = os.getcwd()

    if name is None:
        name = f"plot_{datetime.now()}"

    if format == 'html':
        fig.write_html(f"{path}{name}.html")

    if format in ['png', 'svg']:
        fig.write_image(f"{path}{name}.{format}", **kwargs)

    if isinstance(show_browser, str):
        show_browser = eval(show_browser)

    if show_browser:
        fig.show()


def observation_box(data):
    fig = px.box(pd.melt(data), x="variable", y="value")

    fig.update_layout(
        title={
            'text': f"<b>Distribution of number of observation per 'time-series'feature</b>",
            'font': {'size': 25}
        },

        xaxis_title="feature",
        yaxis_title=f'number of observation',
        template='plotly_white'
    )

    return fig


def ECDF_customized(data):
    fig = go.Figure()
    for column in data:
        fig.add_scatter(x=np.unique(data[column]), y=1 - ECDF(data[column])(np.unique(data[column])), line_shape='hv',
                        name=column)

    fig.update_layout(
        title={
            'text': f"<b>Fraction of usable observations per 'time-series'-feature",
            'font': {'size': 25}
        },

        xaxis_title="Feature",
        yaxis_title=f'fraction of usable observations',
        template='plotly_white'
    )
    return fig


def fig_timeseries(filtered_data, item, means, sd):
    fig = px.line(filtered_data,
                  x='variable',
                  y='value',
                  color='final_result',
                  line_group="ID",
                  color_discrete_sequence=general['COLORS_GROUPING'][0:2])

    # add grouped mean/std
    for idx, row in enumerate(means.iterrows()):
        y = list(row[1].values)
        x = list(row[1].index)

        sd_row = list(sd.loc[row[0]].values)
        y_upper = [sum(x) for x in zip(y, sd_row)]

        y_lower = []
        zip_object = zip(y, sd_row)
        for y_i, sd_row_i in zip_object:
            y_lower.append(y_i - sd_row_i)

            # add upper (mean+sd) trace (invisible line)
        fig.add_trace(go.Scatter(
            x=x,
            y=y_upper,
            fill='none',
            mode='lines',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            name=f"y_upper_sd_{row}"
        ))

        # add lower (mean-sd) trace (line invisible but filled area between upper and lower line)
        fig.add_trace(go.Scatter(
            x=x,
            y=y_lower,
            fill='tonexty',
            mode='lines',
            fillcolor=get_color_transparency(general['COLORS_GROUPING'][idx], 0.1),
            line_color='rgba(255,255,255,0)',
            showlegend=True,
            name=f"sd_{row[0]}"
        ))

        fig.add_trace(go.Scatter(x=x, y=y, name=f"mean {row[0]}",
                                 line=dict(width=8, dash='dash', color=general['COLORS_GROUPING'][idx])))

    fig.update_layout(
        title={
            'text': f"<b>grouped timeseries '{item}'</b>",
            'font': {'size': 25}
        },
        xaxis_title="time",
        yaxis_title=f"value",
        template='plotly_white'
    )
    return fig


def fig_feature_importance(feature_importance, classifier):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=feature_importance['Feature Importance'],
                         y=feature_importance['Feature Label'],
                         name='feature_performance',
                         orientation='h',
                         marker=dict(
                             color=general['COLORS']['importance'],
                             line=dict(
                                 color=general['COLORS']['importance'],
                                 width=2
                             )
                         )))
    fig.update_layout(
        title={
            'text': f'<b>Feature Importance: classification algorithm = {type(classifier).__name__}</b>',
            'font': {'size': 25}
        },

        xaxis_title="feature importance value",
        template='plotly_white'
    )

    return fig


def fig_performance_curve(feature_selector, mode='box', boxpoints=False, best_score_line=True):
    """

    :param feature_selector:
    :param mode:
    :param boxpoints: False (default), 'all', 'outliers', 'suspectedoutliers'
    :return:
    """

    cv_results_rfe = pd.DataFrame(feature_selector.cv_results_)
    no_features = feature_selector.n_features_in_
    n_splits = feature_selector.n_splits_

    fig = go.Figure()

    if mode == 'box':

        train_res, cv_res = get_train_cv_data(cv_results_rfe, n_splits)
        x_box, y_train, y_cv = reshape_data_frame(train_res, cv_res, no_features, n_splits)

        y = [y_train, y_cv]
        names = ['train', 'cv']

        for idx, item in enumerate(y):
            fig.add_trace(go.Box(
                y=item,
                x=x_box,
                name=names[idx],
                boxmean='sd',
                boxpoints=boxpoints,
                marker_color=general['COLORS'][names[idx]]
            ))
    if mode == 'point':

        x = list(range(1, no_features + 1))
        y = [cv_results_rfe['mean_train_score'].values, cv_results_rfe['mean_test_score'].values]
        y_err = [cv_results_rfe['std_train_score'], cv_results_rfe['std_test_score']]
        names = ['train', 'cv']

        for idx, item in enumerate(y):
            fig.add_trace(go.Scatter(
                y=item,
                x=x,
                name=names[idx],
                error_y=dict(
                    type='data',  # value of error bar given in data coordinates
                    array=y_err[idx],
                    visible=True),
                marker_color=general['COLORS'][names[idx]]
            ))

    # add vertical line with best cv test score
    if best_score_line:
        no_features_best_score = cv_results_rfe['mean_test_score'].idxmax() + 1  # number of features
        min_value = get_train_cv_data(cv_results_rfe, n_splits)[1].min().min()
        max_value = get_train_cv_data(cv_results_rfe, n_splits)[0].max().max()

        fig.add_trace(go.Scatter(x=[no_features_best_score, no_features_best_score],
                                 y=[min_value * 0.98,
                                    max_value * 1.02],
                                 name="best mean cv-score",
                                 line=dict(color='royalblue', width=4, dash='dot')))

    fig.update_layout(
        title={
            'text': f'<b>{feature_selector.scoring}: depending on number of features '
                    f'(best mean cv-score: {round(feature_selector.best_score_, 5)})</b>',
            'font': {'size': 25}
        },

        xaxis_title="Number of Features",
        yaxis_title=f'score  [{feature_selector.scoring}]',
        template='plotly_white',
        boxmode='group'  # group together boxes of the different traces for each value of x
    )

    return fig


def get_train_cv_data(cv_results_rfe, n):
    headers = cv_results_rfe.columns.to_list()
    idx = [idx for idx, item in enumerate(headers) if item == 'split0_test_score']
    cv_res = cv_results_rfe.iloc[:, idx[0]:(idx[0] + n)]
    idx = [idx for idx, item in enumerate(headers) if item == 'split0_train_score']
    train_res = cv_results_rfe.iloc[:, idx[0]:(idx[0] + n)]
    return [train_res, cv_res]


def reshape_data_frame(train_res, cv_res, no_features, n):
    y_cv = np.reshape(cv_res.values, cv_res.values.size).tolist()  # 1d array
    y_train = np.reshape(train_res.values, train_res.values.size).tolist()  # 1d array
    x = sorted(list(range(1, no_features + 1)) * n)
    x = list(map(str, x))
    return [x, y_train, y_cv]


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def store_results(timestamp_sim, timestamp):
    if os.path.exists(f"{general['store_infos']['location']}/simulation_{timestamp_sim}") is False:
        try:
            os.mkdir(f"{general['store_infos']['location']}/simulation_{timestamp_sim}")
        except OSError:
            print(f"Creation of the directory '{general['store_infos']['location']}/simulation_{timestamp_sim}' failed")

    if os.path.exists(f"{general['store_infos']['location']}/simulation_{timestamp_sim}/{timestamp}") is False:
        try:
            os.mkdir(f"{general['store_infos']['location']}/simulation_{timestamp_sim}/{timestamp}")
        except OSError:
            print(f"Creation of the directory '{general['store_infos']['location']}/simulation_{timestamp_sim}/"
                  f"{timestamp}' failed")

    copytree("data/output",
             f"{general['store_infos']['location']}/simulation_{timestamp_sim}/{timestamp}")

    copyfile("config.py", f"{general['store_infos']['location']}/simulation_{timestamp_sim}/{timestamp}/config.py")
    copyfile("main.py", f"{general['store_infos']['location']}/simulation_{timestamp_sim}/{timestamp}/main.py")


# only results and logs without train-, test-data or raw-data
def store_results_cloud(timestamp_sim, timestamp):
    if os.path.exists(f"{general['store_infos']['location']}/simulation_{timestamp_sim}") is False:
        try:
            os.mkdir(f"{general['store_infos']['location']}/simulation_{timestamp_sim}")
        except OSError:
            print(f"Creation of the directory '{general['store_infos']['location']}/simulation_{timestamp_sim}' failed")

    if os.path.exists(f"{general['store_infos']['location']}/simulation_{timestamp_sim}/[{timestamp}_cloud") is False:
        try:
            os.mkdir(f"{general['store_infos']['location']}/simulation_{timestamp_sim}/{timestamp}_cloud")
        except OSError:
            print(f"Creation of the directory '{general['store_infos']['location']}/simulation_{timestamp_sim}/"
                  f"{timestamp}_cloud' failed")

    copytree("data/output",
             f"{general['store_infos']['location']}/simulation_{timestamp_sim}/{timestamp}_cloud")

    # remove sensitive folders and files:
    for item in general['store_infos']['sensitive_files']:
        try:
            file_path = f"{general['store_infos']['location']}/simulation_{timestamp_sim}/{timestamp}_cloud/{item}"
            os.remove(file_path)
        except FileNotFoundError:
            continue

    copyfile("config.py", f"{general['store_infos']['location']}/simulation_{timestamp_sim}/{timestamp}/config.py")
    copyfile("main.py", f"{general['store_infos']['location']}/simulation_{timestamp_sim}/{timestamp}/main.py")


def _initialize_logger(script):
    path = os.path.join(get_wd(), general['output_path'], 'logger')
    logging, console = initialize_logger(script, path=path)
    logging.info("logger initialized...\n")
    return logging, console


# def setup(script):
#     logging, console = _initialize_logger(script)
#     return logging, console


# def clear_output():
#     # remove logs from folder output
#     shutil.rmtree(general['output_path'])


# @log()
def get_train_test_files(filenames, error_message='fileNotFound'):
    files = []

    for idx, item in enumerate(filenames):
        try:
            file = pd.read_pickle(f"{general['output_path']}{filenames[idx]}.pkl")
            files.append(file)

        except FileNotFoundError:
            pass
            # if logging:
            #     logging(error_message)
            return

    # if logging:
    #     logging.info(f"GET DATA\n")
    #     logging.info(f"successfully imported train- and test-data\n")
    #     logging.info(
    #         f"{filenames[0]}: {files[0].shape}, {filenames[1]}: {files[1].shape}, "
    #         f"{filenames[2]}: {files[2].shape}, {filenames[3]}: {files[3].shape}\n")

    return files[0], files[1], files[2], files[3]
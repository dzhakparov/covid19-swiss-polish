from config import general, evaluation, evaluation_data
from plotly.subplots import make_subplots
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from src.helpers import store_figure
import pandas as pd
from joblib import dump, load
from src.helpers import get_train_test_files, get_console


def run():

    # script = str.split(os.path.basename(__file__), '.')[0]
    # logging, console = setup(script)
    # logging.info(f"parameters from 'ml_params' ('config.py'):\n\n {pformat(ml_params)}\n")
    fitted_models = load(f"{general['output_path']}fitted_models.joblib")

    X_train_ml, X_test_ml, y_train_ml, y_test_ml = get_train_test_files(
        ['X_train_ml', 'X_test_ml', 'y_train', 'y_test'],
        error_message=f"file not found! Run 'ml_pipeline_sax' first to create train- and test-files and "
                      f"'ml_pipeline_transformation' to generate 'X_train_ml' and 'X_test_ml'")

    scorers = []

    console = get_console()
    console.info(f"running evaluation")

    for item in evaluation_data['data']:

        preds = Predictions(fitted_models=fitted_models)

        data = eval(f"X_{item}_ml")
        preds.predict(data)
        # logging.info(f"\n\nprediction {item}: \n{pformat(preds.get_predictions(len=10))}\n")

        true_values = eval(f"y_{item}_ml")
        sc = Scorer(true_values, preds, item, **evaluation)
        sc.run()
        scorers.append(sc)
        # logging.info(f"\n\nscores {item}:\n{pformat(sc.get_scores())}\n")

        cm = ConfusionMatrix(sc)
        cm.run()
        fig = cm.plot(split=item)
        store_figure(fig, name=f'confusion_matrix_{item}', path=f"{general['output_path']}fig/",
                     format=general['image_format'],
                     show_browser=general['show_browser'])
        # logging.info(f"stored figure 'confusion_matrix_{item}' in {general['output_path']}fig \n")
        # logging.info(f"\n\nconfusion matrix {item}: {cm}\n")

    # fig = fig_evaluation(scorers, sort='metric', y_range=[0, 1])
    fig = fig_evaluation(scorers, sort='algorithm', y_range=[0, 1])
    store_figure(fig, name=f"evaluation_algorithm", path=f"{general['output_path']}fig/",
                 format=general['image_format'],
                 show_browser=general['show_browser'])
    # logging.info(f"stored figure 'evaluation_algorithm' in {general['output_path']}fig.\n")

    fig = fig_evaluation(scorers, sort='metric', y_range=[0, 1])
    store_figure(fig, name=f"evaluation_metric", path=f"{general['output_path']}fig/",
                 format=general['image_format'],
                 show_browser=general['show_browser'])
    # logging.info(f"stored figure 'evaluation_metric' in {general['output_path']}fig.\n")

    # build pandas with overview from scorers
    df_results = build_df_overview(scorers)
    # logging.info(f"overview results: \n\n{pformat(df_results)}\n")
    dump(df_results, f"{general['output_path']}df_results.joblib")
    df_results.to_csv(f"{general['output_path']}df_results.csv", index=False)


class Predictions:

    def __init__(self, fitted_models):
        self.fitted_models = fitted_models
        self.type = None
        self.predictions = {}
        self.predictions_short = {}

    def predict(self, X, type=None):
        self.type = type

        for item in self.fitted_models.keys():
            if type == 'proba':
                preds = self.fitted_models[item].predict_proba(X)
            else:
                preds = self.fitted_models[item].predict(X)
            self.predictions.update({item: preds})

    def get_predictions(self, len=None):
        if len is not None:
            for item in self.predictions:
                short_preds = self.predictions[item][0:len]
                self.predictions_short.update({item: short_preds})
            return self.predictions_short
        else:
            return self.predictions


class Scorer:

    def __init__(self, true_values, predictions, name, scoring_methods=['accuracy_score'], args=None):
        self.true_values = true_values
        self.predictions = predictions
        self.name = name
        self.scoring_methods = scoring_methods
        self.args = args
        self.scores = {}
        self.scoring_algorithms = []

    def run(self):

        for item in self.predictions.predictions.keys():
            self.scoring_algorithms.append(item)
            y_pred = self.predictions.predictions[item]
            result = {}
            add = {}
            for method in self.scoring_methods:
                if method in self.args:
                    add = self.args[method]
                scoring_method = getattr(metrics, method)
                score = scoring_method(self.true_values.to_numpy(), y_pred, **add)
                result.update({method: score})
            self.scores.update({item: result})

    def get_scores(self):
        return self.scores


class ConfusionMatrix:

    def __init__(self, scorer):
        self.scorer = scorer
        self.matrix = {}

    def run(self, algorithm='all'):
        """
        :param algorithm: string or list of strings with valid algorithm names of which confusion matrix should be calculated
        :return: dict with confusion matrix as pandas DataFrame
        """
        for item in self.scorer.predictions.predictions.keys():
            if item in algorithm or algorithm == 'all':

                labels = self.scorer.predictions.fitted_models[item].classes_
                conf_matrix = confusion_matrix(self.scorer.true_values.to_numpy(),
                                               self.scorer.predictions.predictions[item],
                                               labels=labels)
                conf_matrix = pd.DataFrame(conf_matrix, columns=list(labels), index=list(labels))
                self.matrix.update({item: conf_matrix})

    def __str__(self):
        string = ""
        for item in self.matrix:
            string = string + f"\n{item}:\n{self.matrix[item]}\n"
        return string

    def get_matrix(self):
        return self.matrix

    def _build_subtitles(self):
        subtitles = []
        for item in self.matrix:
            scores = self.scorer.scores[item]
            score_strings = []
            for key, score in scores.items():
                score_string = f"{key.replace('_score', '')} =  {round(score, 4)}"
                score_strings.append(score_string)
            score = ', '.join(score_strings)
            subtitle = f"<b>{item}</b> <br> {score}"
            subtitles.append({'text': subtitle})
        return subtitles

    def plot(self, split="default", **kwargs):

        no_plots = len(self.matrix)
        positions, geometry = _build_positions(no_plots, **kwargs)

        fig = make_subplots(rows=geometry[0],
                            cols=geometry[1],
                            start_cell="top-left",
                            subplot_titles=list(self.matrix.keys())
                            )

        for idx, (key, data) in enumerate(self.matrix.items()):
            z = data.values
            labels = data.columns.to_list()

            fig.add_trace(go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                colorscale='ylgn',
                showscale=False,
            ),
                row=positions[idx][0], col=positions[idx][1])

        fig.update_yaxes(dict(title='true value', autorange='reversed'))
        fig.update_xaxes(dict(title='predicted value'))

        fig.update_layout(title_text=f'<i><b>Confusion matrix: {split.upper()}</b></i>')
        subtitles = self._build_subtitles()
        fig.update_layout({'annotations': subtitles})
        return fig


def fig_evaluation(scorer, sort='algorithm', **kwargs):
    """
    plots one or multiple Scorer-objects
    :param scorer: Object of class Scorer or list of Objects of class Scorer
    :param sort: one of 'algorithm' or 'metric'
    :param kwargs: y_range as list [0,1], geometry as tuple (2,2)
    :return:
    """
    scorer = _check_scorer_list(scorer)

    # set colors
    colors = []
    for item in scorer:
        if item.name.lower() in general['COLORS'].keys():
            colors.append(general['COLORS'][item.name])  # TODO: define colors in case of false

    # define y-range of plot
    if 'y_range' not in kwargs:
        y_range = [0, 1]  # default
    else:
        if (isinstance(kwargs['y_range'], list) and len(
                kwargs['y_range']) == 2 and 0 <= kwargs['y_range'][0] < kwargs['y_range'][1] <= 1):
            y_range = kwargs['y_range']
        else:
            y_range = [0, 1]

    axis = [scorer[0].scoring_algorithms, scorer[0].scoring_methods]

    if sort == 'metric':
        axis[0], axis[1] = axis[1], axis[0]  # swap

    no_plots = len(axis[0])
    positions, geometry = _build_positions(no_plots, **kwargs)

    fig = make_subplots(rows=geometry[0],
                        cols=geometry[1],
                        start_cell="top-left",
                        subplot_titles=axis[0])

    for index, value in enumerate(axis[0]):
        for idx, item in enumerate(scorer):
            x = axis[1]
            if sort == 'algorithm':
                y = list(item.scores[value].values())
            else:
                y = [item.scores[i][axis[0][index]] for i in item.scores]

            fig.add_trace(go.Bar(name=f"{item.name} {value}", x=x, y=y,
                                 marker_color=colors[idx]),
                          row=positions[index][0], col=positions[index][1])
            fig['layout'][f'yaxis{idx + 1}'].update(range=y_range, autorange=False)

    fig.update_layout(
        title={
            'text': f"<b>Predictions with different Classifiers ({sort})</b>",
            'font': {'size': 25}
        },
        template='plotly_white'
    )

    return fig


def build_df_overview(scorer):
    scorer = _check_scorer_list(scorer)

    results = []
    algorithms = list(scorer[0].scores.keys())
    columns = [['name'], ['split'], scorer[0].scoring_methods]
    columns = [item for sublist in columns for item in sublist]

    for item in scorer:
        values = []
        for algorithm in algorithms:
            values.append(list(item.scores[algorithm].values()))

        df = pd.DataFrame.from_records(values, columns=scorer[0].scoring_methods)
        df['name'] = algorithms
        df['split'] = item.name
        results.append(df)

    df = pd.concat(results)
    df = df[columns]
    return df


def _build_positions(no_plots, **kwargs):
    geometries = [{1: (1, 1)}, {2: (2, 1)}, {3: (2, 2)}, {4: (2, 2)}, {5: (3, 2)}, {6: (3, 2)}, {7: (3, 3)},
                  {8: (3, 3)}, {9: (3, 3)}, {10: (4, 3)}, {11: (4, 3)}, {12: (4, 3)}]

    if 'geometry' not in kwargs:
        geometry = geometries[no_plots - 1].get(no_plots)
    else:
        geometry = kwargs['geometry']
        if geometry[0] + geometry[1] < no_plots:
            geometry = geometries[no_plots - 1].get(no_plots)

    positions = []
    for i in range(geometry[0]):
        for j in range(geometry[1]):
            positions.append((i + 1, j + 1))
    return [positions, geometry]


def _check_scorer_list(scorer):
    if isinstance(scorer, Scorer):
        scorer = [scorer]
    else:
        if isinstance(scorer, list):
            validation = [isinstance(item, Scorer) for item in scorer]
            if not all(validation):
                raise ValueError('Parameter should be a Scorer or a list of Scorer')
        else:
            raise ValueError('Parameter should be a Scorer or a list of Scorer')
    return scorer


if __name__ == '__main__':
   run()
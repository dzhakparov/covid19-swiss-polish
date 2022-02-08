import os
import re
import copy
from itertools import product
from pprint import pformat
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from saxpy.paa import paa
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize

from config import general, sax_params, sax_plot, sax_scan
from src.helpers import store_figure, log, initialize_logger


@log()
def run():

    scaler = sax_params['scaler']
    n_letters = sax_params['n_letters']
    n_length = sax_params['n_length']
    features = sax_params['features']
    # binary_coded = sax_params['binary_coded']
    split_by = sax_params['split_by']
    target = sax_params['target']
    plot_modes = sax_plot['plot_modes']
    show_browser = sax_plot['show_browser']
    plot_selected = sax_plot['plot_selected']
    y = sax_plot['y']
    scan_threshold = sax_scan['warnings_threshold']
    scan_n = sax_scan['warnings_n']

    file = pd.read_pickle(f"{general['output_path']}working_file.pkl")

    # if binary_coded:  # only 2 outcomes: 'died' vs. 'not died'
    #     bool_selection = file['final_result'].str.contains('Death', regex=False)
    #     file['final_result'] = file['final_result'].astype('object')
    #     file.loc[bool_selection, 'final_result'] = 'died'
    #     file.loc[~bool_selection, 'final_result'] = 'not died'

    combinations = list(product(features, n_length, n_letters, scaler))

    obj = SaxTransformer(file, combinations)
    obj.evaluate(target=target, split_by=split_by)  # build evaluation depending on target and split
    # SaxScanner(obj).scan_results(threshold=scan_threshold, n=scan_n)

    plots = SaxPlotter(obj, plot_modes=plot_modes, y=y, selected=plot_selected)
    plots.run(n_jobs=-1, show_browser=show_browser)



class SaxTransformer:

    def __init__(self, file, combinations, id_column='ID'):

        self.file = file
        self.combinations = combinations
        self.ID_column = id_column
        self.sax_coded_objects = []
        self.sax_evaluated_objects = []
        self.split_group = None

        self.logging, self.console = self._initialize_logger()
        self._build_id()
        self._run()

    def __str__(self):
        str = f"SaxTransformator-Object:\n" \
              f"status: {self.get_status()} evaluations (of {self.get_number_evaluations()}) selected"
        return str

    @staticmethod
    def _initialize_logger():
        script = str.split(os.path.basename(__file__), '.')[0]
        logging, console = initialize_logger(script)
        logging.info("logger initialized...\n")
        return logging, console

    @staticmethod
    def _sax_transformation(x, n_letters, n_length):
        val = x.dropna().values
        if len(val) >= 2:
            dat_paa = paa(val, n_length)
            res = ts_to_string(dat_paa, cuts_for_asize(n_letters))
        else:
            res = np.nan
        return res

    @staticmethod
    def _z_transformation(x, mean=None, sd=None):
        val = x.dropna().values

        if len(val) >= 2:
            if mean is None:
                mean = val.mean()
                sd = val.std()

            val = pd.Series(list(np.array(
                [(xi - mean) / sd for xi in x if xi is not np.nan])))  # runtime warning...  -> division by ZERO

            val.index = x.index
            x = val
        return x

    def _build_id(self):
        """ builds a column named 'ID' from existing column (names) or new """
        if self.ID_column not in self.file.columns:
            self.file['ID'] = range(0, self.file.shape[0])
            self.logging.info(f"ID new build: \n {pformat(self.file['ID'].head(5))}")
        else:
            self.file['ID'] = self.file[self.ID_column]
            self.logging.info(f"main-ID build from column '{self.ID_column}': \n\n{pformat(self.file['ID'].head(5))}\n")

    def _normalize_data(self, data, scaler, filter_col):
        # normalisation -> z-norm (per series/row or overall), StandardScaler aso. (per column)
        if scaler == 'None' or scaler is None:
            data_normalized = data
        elif scaler == 'z_all':
            data_reshaped = data.values.reshape((-1, 1))  # all vales in one vector
            data_reshaped = data_reshaped[~np.isnan(data_reshaped)]  # remove missing values
            mean = data_reshaped.mean()  # mean of all values
            std = data_reshaped.std()  # sd of all values
            data_normalized = data.apply(self._z_transformation, mean=mean, sd=std, axis=1)
        elif scaler == 'z_serie':
            data_normalized = data.apply(self._z_transformation, axis=1)
        else:
            data_normalized = scaler.fit_transform(X=data)
            data_normalized = pd.DataFrame(data=data_normalized, columns=filter_col)
        return data_normalized

    def _run(self):
        """ run parallel computations on n_job-cores (-1 = all cores available) """
        self.logging.info(f"run SAX-Transformer: \n\n"
                          f"configurations (features, n_length, n_letters, scaler), no={len(self.combinations)}: "
                          f"\n{pformat(self.combinations)}\n")
        self.sax_coded_objects = Parallel(n_jobs=-1, verbose=1, backend="multiprocessing") \
            (delayed(self._sax)(combination) for combination in self.combinations)
        # dump(self.sax_results, f"{general['output_path']}sax_results.joblib")  # store results
        self.logging.info(f"SAX-transformation successfully completed!\n")

    def _sax(self, combination):
        feature, n_length, n_letters, scaler = combination  # unpack combination tuple:

        filter_col = [col for col in self.file if col.startswith(feature)]  # extract specific columns to transform
        data = self.file[filter_col]
        data_normalized = self._normalize_data(data, scaler, filter_col)

        sax_code = data_normalized.apply(self._sax_transformation, args=(n_letters, n_length), axis=1)
        data['sax_code'] = sax_code
        data_normalized['sax_code'] = sax_code

        return Sax(data, data_normalized, feature, n_length, n_letters, scaler)  # build Sax-Object as result

    def evaluate(self, target=None, split_by=None, **kwargs):
        self.split_group = split_by
        self.logging.info(f"evaluation with: \ntarget = {target}, split_by={split_by}\n")

        for object in self.sax_coded_objects:
            for split in self.split_group:
                obj = copy.deepcopy(object)
                obj.split_group = split
                obj.evaluate(file=self.file, target=target, split_by=split)
                self.sax_evaluated_objects.append(obj)

                for idx, res in enumerate(obj.evaluations):
                    if len(obj.evaluations) == 1:
                        idx=None
                    self.logging.info(f"\ncombination: {obj.get_combination_string(idx=idx)}\n"
                                      f"\n{res}\n\n")

    def get_object_list(self):
        return self.sax_evaluated_objects

    def get_status(self):
        counts = 0
        for item in self.sax_evaluated_objects:
            for evaluation in item.evaluations:
                if item.selected:
                    counts += 1
        return counts

    def reset_status(self):
        for item in self.sax_evaluated_objects:
            for evaluation in item.evaluations:
                item.selected = False

    def get_number_evaluations(self):
        counts = 0
        for item in self.sax_evaluated_objects:
            for evaluation in item.evaluations:
                counts += 1
        return counts


class Sax:

    def __init__(self, data, data_normalized, feature, n_length, n_letters, scaler):

        self.data = data
        self.data_normalized = data_normalized
        self.feature = feature
        self.n_length = n_length
        self.n_letters = n_letters
        self.scaler = scaler
        self.split_group = []
        self.split = []
        self.target = None
        self.selected = False
        self.evaluations = []

        self.plot_mode = None
        self.y = None
        self.features = [feature for feature in self.data.columns if self.feature in feature]
        self.title = ""
        self.store_title = ""
        self.data_plot = None

    def select_item(self):
        self.selected = True

    def unselect_item(self):
        self.selected = False

    def get_combination_string(self, idx=None):
        if idx is None:
            return f"feature: {self.feature}, n_length: {self.n_length}, n_letters: {self.n_letters}, scaler: {self.scaler}, split: {self.split_group}"
        else:
            return f"feature: {self.feature}, n_length: {self.n_length}, n_letters: {self.n_letters}, scaler: {self.scaler}, split: {self.split_group, self.split[idx]}"

    def get_combination(self):
        return self.feature, self.n_length, self.n_letters, self.scaler, self.split_group

    def evaluate(self, file, target=None, split_by=None, **kwargs):
        self._add_additional_infos(file, target, split_by)  # makes copies of object (due split)
        self._stats()  # calculate stats for this feature and combination of parameters

    def _add_additional_infos(self, file, target, split_by):
        for idx, item in enumerate([self.data, self.data_normalized]):
            item['ID'] = file['ID']

            if idx == 0:
                item['group'] = 'original'
            else:
                item['group'] = 'normalized'

            if target in file.columns:
                item['target'] = file[target]
                self.target = target
            else:
                pass

            if split_by != 'None':
                if split_by in file.columns:
                    item['split'] = file[split_by]
            else:
                item['split'] = 'None'

    def _stats(self):
            if self.split_group != 'None':
                subgroups = self.data_normalized['split'].unique().to_list()
                for idx, item in enumerate(subgroups):
                    data = self.data_normalized[self.data_normalized['split'] == item]
                    res = self._calc_stats(data)
                    self.split.append(item)
                    self.evaluations.append(res)
            else:
                data = self.data_normalized
                res = self._calc_stats(data)
                self.evaluations.append(res)

    @staticmethod
    def _calc_stats(data):
        total = data.groupby('target')['ID'].count()
        data = data.dropna(how='all', subset=['sax_code'])
        feature_series = data.groupby(['sax_code', 'target'])['ID'].count()
        grouped_data = data.groupby(['sax_code'])['target'].count()
        percent_data = feature_series.copy()
        total_absolute_data = feature_series.copy()
        total_relative_data = feature_series.copy()
        for counter, (idx, value) in enumerate(feature_series.iteritems()):
            percent_data.iloc[counter] = value / grouped_data[idx[0]]
            total_absolute_data.iloc[counter] = total[idx[1]]
            total_relative_data.iloc[counter] = total[idx[1]] / sum(total)
        result = pd.concat([feature_series, percent_data, total_absolute_data, total_relative_data], axis=1)
        feature = data.columns.to_list()[0].split('_')[0]
        result.columns = [f'abs. {feature}', f'rel. {feature}', 'abs. total', 'rel. total']
        return result


class SaxPlotter:

    def __init__(self, SaxTransformer, plot_modes='timeseries', y='survivalrate', selected=False, **kwargs):

        self.SaxTransformer = SaxTransformer
        self.plot_modes = plot_modes
        self.sax_list = self.SaxTransformer.get_object_list()
        self.y = y
        self.selected = selected
        self.original_data = SaxTransformer.file
        self.selected_combinations = []
        self.updatated_selection = []
        self.style_options = {}
        self._prepare_defaults()
        self._update_style_options(**kwargs)

    def run(self, n_jobs=-1, verbose=1, backend="multiprocessing", **kwargs):
        self._get_combination_to_plot(**kwargs)
        self._add_additional_information(**kwargs)
        Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend) \
            (delayed(self._plot)(sax_obj, self.style_options, **kwargs) for sax_obj in self.updatated_selection)

    def _prepare_defaults(self):
        if not isinstance(self.plot_modes, list):
            self.plot_modes = [self.plot_modes]
        if not isinstance(self.y, list):
            self.y = [self.y]

    def _update_style_options(self, **kwargs):
        self.style_options = {'line_break_after': 120,
                              'font_size_title': 25,
                              'font_size_annotation': 14,
                              'shared_yaxis': True}
        if kwargs:
            for key, value in kwargs.items():
                if key in self.style_options:
                    self.style_options.update({key: value})

    def _get_combination_to_plot(self, **kwargs):

        if self.selected is False:
            self.selected_combinations = self.sax_list
        else:
            selected_combinations = []
            for item in self.sax_list:
                if item.selected:  # get combination to plot (selected by SaxScanner)
                    selected_combinations.append((item, item.get_combination()))

            comb = list(set([i[1] for i in selected_combinations]))
            self.selected_combinations = [i[0] for i in selected_combinations if i[1] in comb]

    def _add_additional_information(self, **kwargs):
        for plot_mode in self.plot_modes:
            if plot_mode.title() == 'Xy':
                for y_ in self.y:
                    for item in self.selected_combinations:
                        item = copy.deepcopy(item)
                        item.plot_mode = plot_mode.title()
                        if y_ != 'survivalrate':
                            if y_ in self.original_data.columns:
                                item.data[y_] = self.original_data[y_]  # add additional column of original data
                                item.data_normalized[y_] = self.original_data[y_]
                                item.y = y_
                        else:
                            item.y = y_
                        self.updatated_selection.append(item)
            else:
                for item in self.selected_combinations:
                    item = copy.deepcopy(item)
                    item.plot_mode = plot_mode.title()
                    self.updatated_selection.append(item)

    def _plot(self, sax_obj, style_options, **kwargs):
        # module = importlib.import_module((os.path.basename(__file__)).split(".")[0])  # actual file
        class_ = getattr(self, sax_obj.plot_mode)  # take module instead of self when class Plot, Timeseries and Xy are
        # outside of class SaxPlotter
        class_(sax_obj, style_options, **kwargs)

    class Plot:

        def __init__(self, sax_obj, style_options, show_browser=False, format='html', **kwargs):
            self.sax_obj = sax_obj
            self.annotations = None
            self.grouped_data = None
            self.style_options = style_options
            self.format = format
            self.show_browser = show_browser

        def _build_count_stats(self):
            self.grouped_data = self.sax_obj.data[['target', 'sax_code', 'ID', 'split']].groupby(
                ['sax_code', 'target', 'split']).agg(['count'])
            self.grouped_data.reset_index(inplace=True)
            self.grouped_data.columns = ['sax_code', 'target', 'split', 'counts']
            self.sum_groups = self.grouped_data.groupby(['sax_code', 'split']).sum().reset_index()

        def _build_title(self):
            self.sax_obj.title = f"<b>SAX {self.__class__.__name__}': {self.sax_obj.feature}'</b>" \
                                 f"  (scaler={self.sax_obj.scaler}, n_length={self.sax_obj.n_length}, n_letters={self.sax_obj.n_letters})"

        def _build_store_title(self, y=None):
            if y:
                self.sax_obj.store_title = f"SAX_{self.__class__.__name__}_{y}_{self.sax_obj.feature}_splitted_by-{self.sax_obj.split_group}_{self.sax_obj.scaler}_length-" \
                                           f"{self.sax_obj.n_length}_letters-{self.sax_obj.n_letters}"
            else:
                self.sax_obj.store_title = f"SAX_{self.__class__.__name__}_{self.sax_obj.feature}_splitted_by-{self.sax_obj.split_group}_{self.sax_obj.scaler}_length-" \
                                           f"{self.sax_obj.n_length}_letters-{self.sax_obj.n_letters}"

        def _build_annotations_dict(self):

            target = [str(i) for i in sorted(self.grouped_data['target'].unique())]
            sax_codes = list(self.grouped_data['sax_code'].unique())
            annotations = {}

            if self.sax_obj.split is not None:
                splits = list(self.grouped_data['split'].unique())
            else:
                splits = ['None']

            for split in splits:
                text = []
                total = 0
                for sax_code in sax_codes:
                    data_mod = self.grouped_data[(self.grouped_data["sax_code"] == sax_code) & (
                            self.grouped_data['split'] == split)]
                    data_mod = data_mod.sort_values(by=['target'])
                    counts = list(data_mod['counts'].values)
                    total = total + sum(counts)
                    str_counts = [str(i) for i in counts]
                    temp_text_1 = f"<b>{sax_code}</b>: {'/'.join(str_counts)}"
                    text.append(temp_text_1)
                temp_text = f"({'/'.join(target)}): " + ', '.join(text)
                annotations.update({str(split): f"{temp_text},  <b>TOTAL: {total}</b>"})

            self.annotations = annotations

        @staticmethod
        def add_line_break(line_break_after, string):
            if len(string) > line_break_after:
                substring = ","
                splitted_string = []

                while len(string) > line_break_after:
                    matches = re.finditer(substring, string)
                    matches_positions = [match.start() for match in matches]
                    pos_x = [i for i in matches_positions if i < line_break_after]
                    pos_x = max(pos_x) + 1
                    splitted_string.append(f"{string[0:pos_x]}<br>")
                    string = string[pos_x + 1:]

                splitted_string.append(string)
                string = ''.join(splitted_string)
            return string

    class Timeseries(Plot):
        def __init__(self, sax_obj, style_options, **kwargs):
            super().__init__(sax_obj, style_options, **kwargs)
            self.run()

        def run(self):

            self._build_title()
            self._build_store_title()
            self._data_preparation()
            self._build_count_stats()
            self._build_annotations_dict()
            for key, value in self.annotations.items():
                self.annotations.update({key: self.add_line_break(self.style_options['line_break_after'], value)})
            fig = self._make_fig()

            store_figure(fig, name=f"{self.sax_obj.store_title}", path=f"{general['output_path']}sax/",  # TODO (path)
                         format=self.format, show_browser=self.show_browser)

        def _data_preparation(self):

            self.sax_obj.data_plot = pd.concat([self.sax_obj.data, self.sax_obj.data_normalized], axis=0)
            self.sax_obj.data_plot.dropna(how='all', subset=['sax_code'],
                                          inplace=True)  # remove rows which have less than 2 valid values

            non_features = [i for i in self.sax_obj.data_plot.columns.to_list() if i not in self.sax_obj.features]

            self.sax_obj.data_plot = pd.melt(self.sax_obj.data_plot, id_vars=non_features,
                                             value_vars=self.sax_obj.features)

        def _make_fig(self, **kwargs):

            fig = px.line(self.sax_obj.data_plot,
                          x='variable',
                          y='value',
                          color='sax_code',
                          line_dash='target',
                          line_group="ID",
                          facet_row="group",
                          facet_col="split"
                          )

            for idx, annotation in enumerate(fig.layout.annotations):

                if self.sax_obj.split is None:
                    annotation.text = f"splitted_by: <b> None </b><br> {self.annotations['None']}"
                    annotation.yref = 'paper'
                    annotation.y = 1.10
                    annotation.x = 0.5
                    annotation.textangle = 0
                    annotation.font = dict(size=self.style_options['font_size_annotation'])
                    break
                else:
                    split = annotation.text.split('=')[1]
                    if idx < len(self.annotations):
                        annotation.text = f"splitted by: <b>{self.sax_obj.split_group}: {split}</b><br>{self.annotations[split]}]"
                        annotation.yref = 'paper'
                        annotation.y = 1.05
                        annotation.font = dict(size=self.style_options['font_size_annotation'])

            fig.update_layout(
                title={
                    "text": self.sax_obj.title,
                    "font_size": self.style_options['font_size_title'],
                    "yref": "container",
                    "y": 0.96,
                    "yanchor": "top"
                },
                xaxis_title=f"",
                yaxis_title=f"value",
                template='plotly_white',
                margin=dict(
                    t=250,
                ),
            )

            fig.update_yaxes(matches=None)
            return fig

    class Xy(Plot):
        def __init__(self, sax_obj, style_options, **kwargs):
            super().__init__(sax_obj, style_options, **kwargs)
            self.positive_group = None  # arg
            self.run()

        def run(self):

            y_ = self.sax_obj.y

            self._build_title()
            self._build_store_title(y=y_)

            self._data_preparation()  # like in 'Timeseries' -> to Plot
            self._build_plot_object(y_)  # TODO: ERROR!
            self._build_annotations_dict()
            for key, value in self.annotations.items():
                self.annotations.update({key: self.add_line_break(self.style_options['line_break_after'], value)})
            fig = self._make_fig(y_)

            store_figure(fig, name=f"{self.sax_obj.store_title}", path=f"{general['output_path']}sax/",  # TODO (path)
                         format=self.format, show_browser=self.show_browser)

        def _add_x(self):
            data_temp_store = []
            targets = list(self.grouped_data['target'].unique())
            for target in targets:
                for item in list(set(list(self.grouped_data['split']))):
                    data_temp = self.grouped_data[
                        (self.grouped_data['split'] == item) & (self.grouped_data['target'] == target)]
                    data_temp.sort_values(by='value', inplace=True, ascending=False)
                    data_temp['x'] = list(range(0, data_temp.shape[0]))
                    data_temp_store.append(data_temp)
                    res = pd.concat(data_temp_store)
            return res

        def _data_preparation(self):
            # survivalrate: choose between 'died' and 'not died'
            self.positive_group = sorted(self.sax_obj.data['target'].unique())[0]

        def _build_plot_object(self, y_):

            if y_ != 'survivalrate':
                self.grouped_data = self.sax_obj.data[['target', 'sax_code', 'split', y_]].groupby(
                    ['target', 'sax_code', 'split']).mean()
                self.grouped_data.fillna(0, inplace=True)
                self.grouped_data.reset_index(inplace=True)
                self.grouped_data = pd.melt(self.grouped_data, id_vars=['target', 'sax_code', 'split'], value_vars=y_)
                self.grouped_data = self.grouped_data.sort_values(['target', 'split', 'value'], ascending=False)
                self.sax_obj.data_plot = self._add_x()
                self._build_count_stats()
            else:
                self._build_count_stats()

                def fun_group(row, sum_group, split):
                    sax_code = list(sum_group['sax_code'] == row['sax_code'])
                    sp = list(sum_group[split] == row[split])
                    bool_list = [x & y for (x, y) in zip(sax_code, sp)]
                    value = sum_group[bool_list]['counts'].values
                    return float(row['counts'] / value)

                self.grouped_data['value'] = self.grouped_data.apply(fun_group,
                                                                     args=(self.sum_groups, 'split'),
                                                                     axis=1)

                self.sax_obj.data_plot = self._add_x()
                self.sax_obj.data_plot = self.sax_obj.data_plot[self.sax_obj.data_plot['target'] == self.positive_group]
                self.sax_obj.data_plot.sort_values(by='split', inplace=True, ascending=False)
                self.sax_obj.data_plot = self.sax_obj.data_plot.drop('counts', axis=1)
                self.sax_obj.data_plot['variable'] = 'survivalrate'

        def _make_fig(self, y_):

            COLORS = ["red", "blue", "green", "black"]

            targets = list(self.sax_obj.data_plot['target'].unique())

            if self.sax_obj.split_group != 'split':

                splits = list(self.sax_obj.data_plot['split'].unique())
                ann = []
                for split in splits:
                    ann.append(f"<b>{str(split)}</b><br>{self.annotations[str(split)]}")

                fig = make_subplots(
                    rows=1, cols=len(splits),
                    subplot_titles=(ann),
                    shared_yaxes=self.style_options['shared_yaxis'])

                for idx_split, split in enumerate(splits):
                    for idx_target, target in enumerate(targets):
                        data_mod = self.sax_obj.data_plot[(self.sax_obj.data_plot["target"] == target) & (self.sax_obj.data_plot['split'] == split)]

                        fig.add_trace(go.Scatter(x=data_mod['x'], y=data_mod['value'], marker_color=COLORS[idx_target],
                                                 name=f"{target}_{split}",
                                                 mode="markers+text", text=data_mod['sax_code'],
                                                 textposition="bottom center",
                                                 textfont=dict(
                                                     # family="sans serif",
                                                     size=18,
                                                     # color="crimson"
                                                 )), row=1, col=idx_split + 1)

                        fig.update_xaxes(title=f"sax-categories (sorted by value)", tickfont=dict(size=16),
                                         showline=True,
                                         showticklabels=False, showgrid=True,
                                         linewidth=1, linecolor='black', row=1, col=idx_split + 1)
                        fig.update_yaxes(titlefont={'size': 16}, tickfont=dict(size=16), showline=True, showgrid=False,
                                         linewidth=1, linecolor='black', row=1, col=idx_split + 1)

                        for i in fig['layout']['annotations']:
                            i['font'] = dict(size=self.style_options['font_size_annotation'])
            else:
                fig = make_subplots(
                    rows=1, cols=1)

                for idx, target in enumerate(targets):
                    data_mod = self.sax_obj.data_plot[self.sax_obj.data_plot["target"] == target]
                    fig.add_trace(
                        go.Scatter(x=data_mod['x'], y=data_mod['value'], marker_color=COLORS[idx], name=target,
                                   mode="markers+text+lines", text=data_mod['sax_code'],
                                   textposition="bottom center",
                                   textfont=dict(
                                       # family="sans serif",
                                       size=18,
                                       # color="crimson"
                                   )))

                    fig.update_xaxes(title=f"sax-categories (sorted by value)", tickfont=dict(size=16), showline=True,
                                     showticklabels=False, showgrid=True,
                                     linewidth=1, linecolor='black', row=1, col=1)
                    fig.update_yaxes(titlefont={'size': 16}, tickfont=dict(size=16), showline=True, showgrid=False,
                                     linewidth=1, linecolor='black', row=1, col=1)

                    fig.update_layout(
                        annotations=[dict(xref='paper',
                                          yref='paper',
                                          x=0.5, y=1.1,
                                          showarrow=False,
                                          text=self.annotations['None'])])

                    for i in fig['layout']['annotations']:
                        i['font'] = dict(size=self.style_options['font_size_annotation'])

            if y_ == 'survivalrate':
                yaxis_title = f"{y_} [{self.positive_group}]"
            else:
                yaxis_title = f"{y_} [mean]"

            fig.update_layout(
                title={
                    'text': self.sax_obj.title,
                    'font_size': self.style_options['font_size_title'],
                    'y': 1,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                margin=dict(t=200, l=0),
                template='plotly_white',
                yaxis_title=yaxis_title
            )
            return fig


class SaxScanner:

    def __init__(self, obj):
        self.obj = obj

    @staticmethod
    def _scan_results(results, warnings_threshold, warnings_n, additional):
        warnings = []
        for sax_obj in results:  # results are a collection (list) of SAX-Objects
            for idx, evaluation in enumerate(sax_obj.evaluations):
                if len(sax_obj.split) > 0:
                    name = f"{sax_obj.feature}, n_length: {sax_obj.n_length}, n_letters: {sax_obj.n_letters}, scaler: {sax_obj.scaler}, split: {sax_obj.split_group}: {sax_obj.split[idx]}"
                else:
                    name = f"{sax_obj.feature}, n_length: {sax_obj.n_length}, n_letters: {sax_obj.n_letters}, scaler: {sax_obj.scaler} split: None"
                for idx, item in evaluation.iterrows():
                    if additional is None:
                        if abs(item[1] - item[3]) >= warnings_threshold and item[0] >= warnings_n:
                            warnings.append({name: item})
                            sax_obj.select_item()
                    else:
                        if abs(item[1] - item[3]) >= warnings_threshold and item[0] >= warnings_n and (
                                evaluation.split_group.find(additional) != -1 or
                                evaluation.split.find(additional) != -1 or
                                evaluation.feature.find(additional) != -1):
                            warnings.append({name: item})
                            sax_obj.select_item()
        return warnings

    def scan_results(self, threshold=0.3, n=10, additional=None):

        results = self.obj.get_object_list()
        script = str.split(os.path.basename(__file__), '.')[0] + '_scan'
        logging, console = initialize_logger(script)

        warnings = self._scan_results(results=results,  # add warning at the end of log-file if assumptions are met
                                      warnings_threshold=threshold,
                                      warnings_n=n,
                                      additional=additional)

        logging.info(f"WARNINGS ({len(warnings)}):\n\n"
                     f"settings:\n"
                     f"threshold= {threshold}\n"
                     f"n= {n}\n"
                     f"additional= {additional}"
                     f"\n")

        logging.info(f"\nconfigurations (features, n_length, n_letters, scaler)\n\n")

        for item in warnings:
            for key, value in item.items():
                logging.info(f"\n{key}\n{pformat(value)}\n")


if __name__ == '__main__':
    run()

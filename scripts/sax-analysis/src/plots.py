from config import general, sax_pipe, ml_params
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from misc import MatrixPlotter
from src.helpers import log


@log()
def boxplot_sax_features(X_train, y_train, numeric_feature):

    xtrain = X_train.reset_index()
    ytrain = y_train.reset_index()
    merged_data = pd.merge(xtrain, ytrain, on='index')

    COLORS = {'aa': 'red', 'ab': 'blue', 'ba': 'green', 'bb': 'orange'}

    for item in sax_pipe['sax_groups']:
        item = item[:-1]  # cut-off last sign '_'

        data = merged_data[[numeric_feature, item]]
        grouped_data = data.groupby([item])

        fig = go.Figure()

        for key, value in grouped_data.groups.copy().items():
            if len(list(value)) == 0:
                del grouped_data.groups[key]

        for i in list(grouped_data.groups.keys()):
            fig.add_trace(go.Box(
                y=grouped_data.get_group(i)[numeric_feature],
                name=i,
                marker_color=COLORS[i],
                boxmean='sd',
                boxpoints='all'
            ))

        fig.update_layout(
            title={
                'text': f"feature: {item}",
                'font': {'size': 24}
            },
            template='plotly_white',
        )

        fig.update_yaxes(title_text=numeric_feature)

        # fig.show()
        fig.write_html(f"{general['output_path']}fig/boxplot/boxplot_sax_features_{item}.html")

    # if logging:
    #     logging.info(f"successfully stored 'boxplot_sax_features' in folder {general['output_path']}fig/boxplot/")


@log()
def survivalplot_sax_features(X_train, y_train, positive_group='survived'):

    xtrain = X_train.reset_index()
    ytrain = y_train.reset_index()
    merged_data = pd.merge(xtrain, ytrain, on='index')

    for item in sax_pipe['sax_groups']:
        item = item[:-1]  # cut-off last sign '_'
        data = merged_data[[item, ml_params['target'], 'index']]
        counted_subgroup = data.groupby([item, ml_params['target']]).count()
        counted_subgroup = counted_subgroup.replace(np.nan, 0)
        counted_subgroup['index'] = counted_subgroup['index'].astype(int)

        subtitle = ""
        df = counted_subgroup.reset_index(level=1)
        df = df.pivot(columns='final_result', values='index')
        dict_for_subtitle = df.to_dict(orient='index')

        header = list(dict_for_subtitle[list(dict_for_subtitle.keys())[0]].keys())
        subtitle = subtitle + '(' + '/'.join(header) + ") -> "
        for k, v in dict_for_subtitle.items():
            subtitle = subtitle + ''.join(f"<b>{k}: </b>")
            values = list(v.values())
            values = [str(i) for i in values]
            subtitle = subtitle + '/'.join(values) + "  "

        counted_subgroup['relative'] = counted_subgroup.groupby(level=0).apply(lambda x: x / float(x.sum()))

        survival_rate = counted_subgroup[counted_subgroup.index.get_level_values('final_result') == positive_group]
        survival_rate = survival_rate.sort_values('relative', ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=survival_rate.index.get_level_values(item), y=survival_rate['relative'],
                                 mode='lines+markers',
                                 name='survivalrate'))

        fig.update_layout(
            title={
                'text': f"survivalcurve feature <b>'{item}'</b> <br> {subtitle}",
                'font': {'size': 24}
            },
            template='plotly_white',
            yaxis=dict(range=[0, 1], title_text='survivalrate [%]'),
        )

        fig.update_xaxes(title_text=f"sax-coding '{item}'")
        # fig.show()
        fig.write_html(f"{general['output_path']}fig/survivalcurve/survivalcurve_sax_features_{item}.html")

    # if logging:
    #     logging.info(f"successfully stored 'survivalplot_sax_features' in folder {general['output_path']}fig/survivalcurve/")


def fig_heatmap(corr_matrix, setup):
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.axes[1],
        y=corr_matrix.axes[0],
        hoverongaps=False))

    fig.update_layout(
        title={
            'text': f'<b>feature correlation (target: {setup["target"]}, filter: {setup["corr_filter"]})</b>',
            'font': {'size': 25}
        },
        template='plotly_white'
    )

    return fig


def visualize_matrix(file, sort_by=None):
    """
    :param file: pandas DataFrame
    :param sort_by: string with column name or list with column names
    :return: plotly heatmap-object
    """

    if sort_by is not None:
        if isinstance(sort_by, str) or isinstance(sort_by, list):
            cl = None
            if isinstance(sort_by, list):
                cl = [i for i in file.columns if i in sort_by]
            else:
                if sort_by in file.columns:
                    cl =[sort_by]
            if cl is not None:
                file = file.sort_values(by=cl, ascending=False)
        else:
            raise Exception("sort_by has to be a single column-name as string or a list of column-names")

    # replace non-numeric cells with np.nan
    df = file[file.applymap(isnumber)]

    scaler = MinMaxScaler()
    scaler.fit(df)

    x = df.columns.to_list()
    y = ['i' + str(i) for i in df.index]
    values = scaler.transform(df)

    trace = go.Heatmap(
        x=x,
        y=y,
        z=values,
        # type='heatmap',
        colorscale='RdBu'
    )
    data = [trace]
    fig = go.Figure(data=data)

    return fig


def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False


@log()
def matrix_plots(data):

    fig1 = MatrixPlotter(data, sort_by=[('final_result', 'asc'), ('age', 'asc')]).get_plot()
    fig1.write_html(f"{general['output_path']}fig/MatrixPlot_individual_scaled.html")

    fig2 = MatrixPlotter(data, sort_by=[('final_result', 'asc'), ('age', 'asc')],
                         scale_by=('group', 0)).get_plot()
    fig2.write_html(f"{general['output_path']}fig/MatrixPlot_group_scaled.html")

    fig3 = MatrixPlotter(data, sort_by=[('final_result', 'asc'), ('age', 'asc')],
                         scale_by=('group', 3)).get_plot()
    fig3.write_html(f"{general['output_path']}fig/MatrixPlot_group_scaled_3sd.html")
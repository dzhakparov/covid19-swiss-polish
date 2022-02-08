import os
import time
import pandas as pd
from pandas_profiling import ProfileReport
from config import general, eda
from src.helpers import store_figure, observation_box, ECDF_customized, fig_timeseries, log
from misc.dataframe_converter.column_conversion import type_plotter


@log()
def run():

    script = str.split(os.path.basename(__file__), '.')[0]

    file = pd.read_pickle(f"{general['output_path']}working_file.pkl")

    if eda['to_analyze']:
        data = file.loc[:, eda['to_analyze']]
    else:
        data = None

    # if isinstance(data, pd.DataFrame):
    #     profile_report(data, script)

    observations_per_multi_feature(file)

    visualize_grouped_timeseries(file,  eda['visualize_grouped_timeseries']['features'])

    # custom (own build) type image
    fig = type_plotter(file, sort="not_missing")
    fig.write_html(f"{general['output_path']}fig/type_plot.html")


@log()
def sweetviz(data, path_output):
    # report with sweetviz
    # https://github.com/fbdesignpro/sweetviz
    # https://towardsdatascience.com/powerful-eda-exploratory-data-analysis-in-just-two-lines-of-code-using-sweetviz-6c943d32f34
    my_report = sweetviz.analyze(data, pairwise_analysis='on')
    my_report.show_html(f"{path_output}report.html")


@log()
def profile_report(data, script):
    # report with ProfileReport
    prof = ProfileReport(data)
    prof.to_file(output_file=f"{general['output_path']}report.html")
    # logging.info(f"Report successfully generated! ({general['output_path']}report.html)")
    # console.info(f"{script} successfully finished!")


@log()
def observations_per_multi_feature(file):

    # ###############################################################################
    # #            generate size of observations per multi-feature                  #
    # ###############################################################################

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    features = file.select_dtypes(include=numerics).columns.to_list()
    features = [item for item in features if
                item[-1].isdigit()]  # if last chiffre is a number there are more than one of this feature
    features = list(
        set([item.rsplit('_', 1)[0] for item in features]))  # take all uniques of the name except this digit

    def count(x):
        no_obs = len(x) - x.isnull().sum()
        return no_obs

    dict_count_observations = {}
    for item in features:
        filter_col = [col for col in file if col.startswith(item)]
        data = file[filter_col]
        counts = data.apply(count, axis=1).to_list()
        dict_count_observations.update({f'{item}': counts})

    data = pd.DataFrame(dict_count_observations)
    # logging.info(f"\n\n'distribution of number of time-series'-features:\n{data.describe()}\n")

    fig = ECDF_customized(data)  # https://community.plotly.com/t/plot-the-empirical-cdf/29045/2
    store_figure(fig, name='ECDF_customized', path=f"{general['output_path']}fig/", format=general['image_format'],
                 show_browser=general['show_browser'])
    # store_figure(fig, path=general['docu_images_path'], name='ECDF_customized', format='png', width=500, height=1000)
    # logging.info(f"successfully plotted: {general['output_path']}fig/ECDF_customized\n")

    fig = observation_box(data)
    store_figure(fig, name='observation_box', path=f"{general['output_path']}fig/", format=general['image_format'],
                 show_browser=general['show_browser'])
    # store_figure(fig, path=general['docu_images_path'], name='observation_box', format='png', width=1200, height=800)
    # logging.info(f"successfully plotted: {general['output_path']}fig/observation_box\n")


@log()
def visualize_grouped_timeseries(file, features):

    # ###############################################################################
    # #         plot time-series-features according to group to classify            #
    # ###############################################################################

    if os.path.exists(f"{general['output_path']}/grouped_timeseries") is False:
        try:
            os.mkdir(f"{general['output_path']}/grouped_timeseries")
        except OSError:
            pass
            # logging.info(f"Creation of the directory failed")

    # 2 categories -> died vs. not_died
    bool_selection = file['final_result'].str.contains('Death', regex=False)
    file['final_result'] = file['final_result'].astype('object')
    file.loc[bool_selection, 'final_result'] = 'died'
    file.loc[~bool_selection, 'final_result'] = 'not died'

    for item in features:

            feature = [i for i in file.columns.to_list() if i.startswith(item)]
            all = ['ID', 'final_result'] + feature
            filtered_data = file[all]

            means = filtered_data.groupby('final_result').mean()
            sd = filtered_data.groupby('final_result').std()

            filtered_data = pd.melt(filtered_data, id_vars=['ID', 'final_result'], value_vars=feature)

            fig = fig_timeseries(filtered_data, item, means, sd)
            store_figure(fig, name=f"{item}", path=f"{general['output_path']}grouped_timeseries/", format=general['image_format'],
                         show_browser=general['show_browser'])

    # logging.info(f"successfully created {len(features)} images in {general['output_path']}grouped_timeseries")


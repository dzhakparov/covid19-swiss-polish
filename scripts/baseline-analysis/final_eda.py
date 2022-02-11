import config
import pandas as pd
from pathlib import Path
from joblib import load
from helpers import _build_summary_data, plot_missing_values_per_group
from PredictorPipeline.preprocessing.preprocessor import Preprocessor

from misc.df_plotter.matrix_plotter import MatrixPlotter
from misc.df_plotter.mv_plotter import MissingValuePlotter
from misc.df_plotter.column_plotter import ColumnPlotter
from misc.df_plotter.type_plotter import TypePlotter
from misc.df_plotter.box_plotter import BoxPlotter


# EDA = Exploratory Data Analysis
# modifies preprocessed file -> drops columns and observations to get better base for modeling and prediction

def run():
    pp = load(f"{config.store_path}/preprocessor.joblib")
    pp.get_summary().to_csv(f'{config.store_path}/summary_{config.working_file}.csv')

    # (descriptive) plots before reducing patients or features
    plot_data = pp.get_data()

    fig = ColumnPlotter(df=plot_data).get_plot()
    fig.write_html(f"{config.store_path}/eda_ColumnPlotter.html")

    fig = MissingValuePlotter(df=plot_data).get_plot()
    fig.write_html(f"{config.store_path}/eda_MissingValuePlotter.html")

    fig = TypePlotter(df=plot_data, sort='missing').get_plot()
    fig.write_html(f"{config.store_path}/eda_TypePlotter.html")

    fig = MatrixPlotter(plot_data, sort_by=[('group', 'asc'), ('ferritin_', 'asc')]).get_plot()
    fig.write_html(f"{config.store_path}/eda_MatrixPlotter.html")

    bp = BoxPlotter(df=plot_data, features=None, split='group')
    bp.update_layout(n_cols=3, n_rows=3, style_grid={'vertical_spacing': 0.1, 'horizontal_spacing': 0.1},
                     colors={'positive': 'green', 'negative': 'red'}, style_figure={'boxpoints': 'all'})
    figs = bp.get_plot()
    bp.store(path=config.store_path, name='eda_BoxPlotter')

    data = pd.read_pickle(Path(f"{config.store_path}/{config.working_file}.pkl"))

    # take all usable columns
    data = data.loc[:, config.usable_columns]

    # missing values per group
    data_summary = _build_summary_data(data, config.target)

    fig = plot_missing_values_per_group(data_summary, config.target)
    fig.write_html(f"{config.store_path}/eda_mv_per_group_before_transformation.html")

    # ***** start modifying working_file *********

    new_pp = Preprocessor(df=data, log=True, path=config.store_path)
    new_pp.drop_columns(thresh=config.drop_threshold_subgroup, group=config.target)
    new_pp.get_summary().to_csv(f'{config.store_path}/summary_{config.working_file}_adjusted.csv')
    data = new_pp.get_data()

    # drop 'outliers'
    # data['quot_baso_lymphocytes_'] =  data['quot_baso_lymphocytes_'][data['quot_baso_lymphocytes_'] < 0.57]
    # data['quot_plt_lymphocytes_'] = data['quot_plt_lymphocytes_'][data['quot_plt_lymphocytes_'] < 1850]
    # data['quot_mon_lymphocytes_'] = data['quot_mon_lymphocytes_'][data['quot_mon_lymphocytes_'] < 6.8]
    # data['quot_eos_lymphocytes_'] = data['quot_eos_lymphocytes_'][data['quot_eos_lymphocytes_'] < 6.0]
    # data['quot_neu_lymphocytes_'] = data['quot_neu_lymphocytes_'][data['quot_neu_lymphocytes_'] < 60]
    # data['quot_plt_neu_'] = data['quot_plt_neu_'][data['quot_plt_neu_'] < 400]

    # ***** end modifying working_file *********

    data_summary = _build_summary_data(data, config.target)

    fig = plot_missing_values_per_group(data_summary, config.target)
    fig.write_html(f"{config.store_path}/eda_mv_per_group_after_transformation.html")

    # storing (modified) data
    data.to_pickle(f"{config.store_path}/{config.working_file}_adjusted.pkl")
    data.to_csv(f"{config.store_path}/{config.working_file}_adjusted.csv")

    plot_data = new_pp.get_data()

    fig = ColumnPlotter(df=plot_data).get_plot()
    fig.write_html(f"{config.store_path}/eda_ColumnPlotter_adjusted.html")

    fig = MissingValuePlotter(df=plot_data).get_plot()
    fig.write_html(f"{config.store_path}/eda_MissingValuePlotter_adjusted.html")

    fig = TypePlotter(df=plot_data, sort='missing').get_plot()
    fig.write_html(f"{config.store_path}/eda_TypePlotter_adjusted.html")

    fig = MatrixPlotter(plot_data, sort_by=[('group', 'asc'), ('ferritin_', 'asc')]).get_plot()
    fig.write_html(f"{config.store_path}/eda_MatrixPlotter_adjusted.html")

    bp = BoxPlotter(df=plot_data, features=None, split='group')
    bp.update_layout(n_cols=3, n_rows=3, style_grid={'vertical_spacing': 0.1, 'horizontal_spacing': 0.1},
                     colors={'positive': 'green', 'negative': 'red'}, style_figure={'boxpoints': 'all'})
    figs = bp.get_plot()
    bp.store(path=config.store_path, name='eda_BoxPlotter_adjusted')
    print(f"final_eda finished!")


if __name__ == '__main__':
    run()

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
import os

# EDA = Exploratory Data Analysis
# modifies preprocessed file -> drops columns and observations to get better base for modeling and prediction


def run_final_eda():

    if os.path.exists(Path(config.store_path, 'plots')) is False:
        try:
            os.mkdir(Path(config.store_path, 'plots'))
        except FileNotFoundError:
            raise Exception(f"Could not build directory!")
    
    # (descriptive) plots before reducing patients or features
    plot_data = pd.read_pickle(Path(f"{config.store_path}/{config.working_file}.pkl"))

    # fig = ColumnPlotter(df=plot_data).get_plot()
    # fig.write_html(f"{config.store_path}/ColumnPlotter.html")

    fig = MissingValuePlotter(df=plot_data).get_plot()
    fig.write_html(f"{config.store_path}/MissingValuePlotter.html")

    fig = TypePlotter(df=plot_data, sort='missing').get_plot()
    fig.write_html(f"{config.store_path}/TypePlotter.html")

    # fig = MatrixPlotter(plot_data, sort_by=[('group', 'asc'), ('ferritin_', 'asc')]).get_plot()
    # fig.write_html(f"{config.store_path}/MatrixPlotter.html")

    bp = BoxPlotter(df=plot_data, features=None, split=config.target)
    bp.update_layout(n_cols=3, n_rows=3, style_grid={'vertical_spacing': 0.1, 'horizontal_spacing': 0.1},
                     colors={'survived': 'green', 'died': 'red'}, style_figure={'boxpoints': 'all'})
    figs = bp.get_plot()
    bp.store(path=config.store_path, name='BoxPlotter')

    data = pd.read_pickle(Path(f"{config.store_path}/{config.working_file}.pkl"))

    # take all usable columns
    # data = data.loc[:, config.usable_columns]

    # missing values per group
    data_summary = _build_summary_data(data, config.target)

    fig = plot_missing_values_per_group(data_summary, config.target)
    fig.write_html(f"{config.store_path}/mv_per_group.html")
    print("EDA finished!")


if __name__ == '__main__':
    run_final_eda()

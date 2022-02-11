import os
import config
import pandas as pd
import numpy as np
from pathlib import Path
from helpers import repl_text_with_initial_number, replace_cell_content_strings, build_quotient_ts
from PredictorPipeline.preprocessing.preprocessor import Preprocessor


def run_preprocessing_longitudinal():

    if os.path.exists(config.store_path) is False:
        try:
            os.mkdir(config.store_path)
        except FileNotFoundError:
            raise Exception(f"Could not build directory!")

    renamed_files = []

    for item in zip(config.data_files, config.renaming_files):
        file = pd.read_csv(item[0])
        rn = pd.read_csv(item[1])

        rn['key'] = file.columns
        rn_dict = dict(zip(rn['key'], rn['working_title']))
        rn_dict = {k: v for k, v in rn_dict.items() if v is not np.nan}

        pp = Preprocessor(df=file, path=config.store_path, log=True)
        pp.rename_columns(rename_dict=rn_dict)
        cols = [i for i in pp.df_actual.columns if i not in rn_dict.values()]  # empty columns
        pp.drop_columns(cols=cols)
        renamed_files.append(pp.df_actual)

    # merge files by 'ID' and drop columns existing in multiple files (base: lab_17_08.csv)
    df = None
    for idx, file in enumerate(renamed_files):
        if idx != 0:
            remove_duplicated_cols = [i for i in file.columns if i in df.columns and i != 'ID']
            file.drop(columns=remove_duplicated_cols, axis=1, inplace=True)
            file.set_index('ID', inplace=True)
            df = df.join(file)
        else:
            df = file.set_index('ID')

    df = df.reset_index()
    df = df[~df['ID'].isnull()]  # drops rows if 'ID' is np.nan (came from files with additional rows)

    # OUT OF CLASS (not logged) -> implement in Preprocessor later...
    df['final_result'] = df['final_result'].apply(
        lambda x: config.recoding_final_results[x] if x in config.recoding_final_results else None)

    df.loc[:, config.repl_cell_content_strings] = df.loc[:, config.repl_cell_content_strings].apply(
        replace_cell_content_strings, new_value=1, axis=1)

    # mixed number (first) and text -> will be replaced only by a new value (text is dropped)
    df = repl_text_with_initial_number(df, config.repl_text_with_number)

    for item in config.outliers:
        df.loc[df['ID'] == item[0], item[1]] = np.nan

    # all Dataframe are renamed und joined to one DataFrame -> this Dataframe can be proceed by 'Preprocessor'
    prepr = Preprocessor(df=df, path=config.store_path)
    prepr.logger.set_logging_options(max_rows=500, max_width=400, max_colwidth=90)  # for displaying DataFrame
    prepr.drop_columns(cols=config.drop_columns)
    prepr.replace_values(pattern=config.repl_pattern, value=config.repl_pattern_values)
    prepr.combine_features(combine_dict=config.combine_features, mode='binary', drop=True)
    prepr.convert_column_type(columns_dict=config.column_conversion_dict)

    # There are almost only sax_features with more than 50% missing values...
    # drop columns that exceed a certain fraction of missing  values and that are NOT a sax_feature
    df = prepr.get_data()
    drop_cols = df.loc[:, df.isnull().mean() > config.drop_threshold].columns.to_list()
    drop_cols = [i for i in drop_cols if not i.startswith(config.sax_features)]
    prepr.drop_columns(cols=drop_cols)

    df = prepr.get_data()

    # OUT OF CLASS (not logged) -> implement in Preprocessor later ...
    df = build_quotient_ts(df, config.build_quotient_ts_cols, drop=True)

    # store pickled preprocessed df
    try:
        file = Path(config.store_path).joinpath(config.working_file)
        df.to_csv(f"{file}.csv", index=False)
        df.to_pickle(f"{file}.pkl")

    except FileNotFoundError:
        raise Exception(f"path '{config.store_path}' does not exists!")
    print("Preprocessing finished!")


if __name__ == '__main__':
    run_preprocessing_longitudinal()

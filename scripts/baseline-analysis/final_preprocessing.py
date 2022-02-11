import os
import itertools
import pandas as pd
import config
from PredictorPipeline.preprocessing.preprocessor import Preprocessor
from helpers import read_in_files, get_first_value, replace_text_with_number


def run():
    if os.path.exists(config.store_path) is False:
        try:
            os.mkdir(config.store_path)
        except FileNotFoundError:
            raise Exception(f"File not found!")

    files = read_in_files(filenames=config.data_files)

    # FILE 0: positive patients
    files[0] = files[0].iloc[:, config.columns_id]
    files[0].columns = config.columns_names
    files[0] = files[0].apply(get_first_value, args=[config.first_value],
                              axis=1)  # if feature_1 has no value but feature_2 has a value, this value will be taken
    files[0]['group'] = 'positive'  # add group to patients
    files[0] = files[0].drop(config.remove_cols_file1, axis=1)
    files[0] = files[0].iloc[0:201, :]  # remove rows after last patient in file

    # FILE 1: negative patients
    files[1] = files[1].rename(columns=config.new_names)
    files[1].columns = [i.lower() for i in files[1].columns.to_list()]
    files[1] = files[1].drop(config.remove_cols_file2, axis=1)
    match = {'Wyleczony': 'Recovered', 'Death negative': 'Death',
             'Negative': 'Negative'}  # adding final result / translating
    files[1]['final_result'] = files[1]['final_result'].apply(lambda x: match[x] if x in match else None)
    files[1]['group'] = 'negative'

    # merge files vertically
    working_file_pn = pd.concat([files[0], files[1]], axis=0)

    # add/merge files[2] (appendix) with id
    # ATTENTION: Third file (appendix) was build by hand from the 2 files "
    #            (CoV-2_negative_final_lab_10_29_appendix and CoV-2_positive_final_lab_10_27_appendix).
    #            4 first values in features have been replaced by the second value of the same feature if the first
    #            value was missing or n/a (positive file).
    #            All additional columns from positive-file have been removed manually and all renamings have been
    #            made manually too in both files.
    #            After this: Merge (over id) with the other 2 files have been successful!
    working_file_pn = pd.merge(working_file_pn, files[2], on='id')

    # replace , to . (decimal)
    columns = working_file_pn.columns.to_list()
    for col in columns:
        working_file_pn[col] = working_file_pn[col].apply(lambda x: str(x).replace(',', '.'))

    # set id as index -> result is a pre-preprocessed file that can be give as base to Preprocessor
    working_file_pn.set_index('id', inplace=True)

    working_file_pn[config.repl_text_with_number] = working_file_pn[config.repl_text_with_number]. \
        apply(replace_text_with_number, axis=1)

    pp = Preprocessor(df=working_file_pn, log=True, path=config.store_path)  # initialize class Preprocessor

    pp.replace_values(pattern=config.replace_values['pattern'], value=config.replace_values['value'],
                      columns=config.replace_values['columns'])

    numerical_columns = tuple(
        [item for item in working_file_pn.columns.to_list() if item not in config.convert_categorical])

    pp.convert_column_type(columns_dict={numerical_columns: ('numeric', ''),
                                         config.convert_categorical: ('category', '')})

    pp.store()

    # drop outliers
    data = pp.get_data()

    # add new (quotient) categories
    for item in config.build_quotient:
        new_name = f"quot_{item[0]}{item[1]}"
        data[new_name] = data.loc[:, item[0]] / data.loc[:, item[1]]
    columns_to_drop = set(itertools.chain.from_iterable(config.build_quotient))
    data = data.drop(list(columns_to_drop), axis=1)  # remove used column

    # data['quot_eos_lymphocytes_'][data['quot_eos_lymphocytes_'] > 2] = np.nan
    # data['quot_mon_lymphocytes_'][data['quot_mon_lymphocytes_'] > 10] = np.nan

    # use only patients with troponin > 0.1
    if config.threshold_troponin[0]:
        if config.threshold_troponin[1] == '>=':
            data = data[data['troponin_'] >= config.threshold_troponin[2]]
        else:
            data = data[data['troponin_'] < config.threshold_troponin[2]]

    pp_1 = Preprocessor(df=data, log=False)
    pp_1.get_data().to_csv(f"{config.store_path}/{config.working_file}.csv", index=True)
    pp_1.get_data().to_pickle(f"{config.store_path}/{config.working_file}.pkl")
    pp_1.get_summary().to_csv(f"{config.store_path}/summary_after_preprocessing.csv", index=False)
    print(f"final_preprocessing finished!")


if __name__ == '__main__':
    run()

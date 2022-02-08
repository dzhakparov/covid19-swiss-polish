import pandas as pd
from functools import reduce
import numpy as np
from config import general
from config import preprocessing as prep
from src.helpers import replace_cell_content_strings, replace_cell_content, log
from pandas.api.types import CategoricalDtype
import math


@log(mode=prep.get('mode'))
def run(files, renaming_files):

    """ runs the whole file preprocessing and returns a preprocessed pandas DataFrame
    :param files: list of files as string
    :param renaming_files: list of renaming files as string
    :return: pre-processed pandas DataFrame
    """

    # remove empty rows (and last few rows with descriptions)
    files, renaming_files = remove_unused_cols_rows(files, renaming_files)

    # rename columns in files with 'working_title'
    files = renaming_columns(files, renaming_files)

    # remove redundant/unimportant columns
    files = remove_columns(files, prep['columns_remove'])

    # merge files
    file,  file_before_preprocessing = merging_files(files)

    # RECODING, TRANSFORMING, ... (real PREPROCESSING)
    file = value_replacement(file)

    file = force_to_numeric(file)

    file = replace_to_nan(file)

    file = repl_text_with_initial_number(file)

    # mixed number (first) and text -> will be replaced only by a new value (text is dropped)
    file[prep['repl_cell_content_strings']] = \
         file[prep['repl_cell_content_strings']].apply(replace_cell_content_strings, new_value=1, axis=1)

    # recode final_result to binary (died vs. survived)
    match = {'Recovered': 'survived', 'Death': 'died', 'Symptoms': 'survived', 'Infected': 'survived'}
    file['final_result'] = file['final_result'].apply(lambda x: match[x] if x in match else None)

    # combine several features to one feature: 1 if at least a single 1 is in the building features, else 0 (or nan)

    def combine_features(x):
        if 1 in x.values:
            return 1
        elif math.isnan(sum(x.values)):
            return np.nan
        else:
            return 0

    for key, value in prep['combine_features'].items():
        file[key] = file.loc[:,value].apply(combine_features, axis=1)
        file.drop(value, axis=1, inplace=True)

    file = convert_column_type(file)

    file = build_quotient_ts(file)

    file = remove_ts(file)

    cols = ['respirator']
    file[cols] = file[cols].replace(np.nan, 'NO_R')

    # outliers:  removing outliers leads to better results in sax (and predicting)
    outliers = [(97,'ldh_2'), (189,'reactive-lympho_2'), (3, 'wbc_3')]
    old_value = []
    for item in outliers:
        old_value.append(file.loc[item[0],item[1]])
        file.loc[item[0], item[1]] = np.nan
    # logging.info(f"manually removed outliers ({len(outliers)} values): \n{pformat(list(zip(outliers, old_value)))}\n")

    # add new feature (categorize)
    file['age_cat'] = pd.qcut(file['age'], q=3)  # x equally sized groups

    # print column types
    pd.set_option("display.max_rows", None)

    file_before_preprocessing.to_csv(f"{general['output_path']}connected_file_before_preprocessing.csv", index=False)
    file.to_csv(f"{general['output_path']}{general['output_filename']}.csv", index=False)
    file.to_pickle(f"{general['output_path']}{general['output_filename']}.pkl")  # for further analysis (stores also column type information)

    return file


@log(mode=prep.get('mode'))
def remove_unused_cols_rows(files, renaming_file):
    """ cut-off columns and rows in data-files (columns) and renaming_files (rows)
    :param files: list of pandas DataFrames
    :return: list of processed pandas DataFrames
    """
    reduced_data_files = []
    reduced_renaming_files = []

    for idx, file in enumerate(files):
        nas = file.isna().sum()
        indices = [idx for idx, item in enumerate(nas) if item == file.shape[0]]  # index is the same for columns in
        # data_files and rows in renaming_files

        file.drop(file.columns[indices], axis=1, inplace=True)  # delete columns
        reduced_data_files.append(
            file.iloc[:prep['last_row_in_bases_files'], ])  # all rows greater than 201 are cut-off (descriptions)

        reduced_renaming_files.append(
            renaming_file[idx].drop(indices, axis=0))  # delete rows in according renaming-file
        # logging.info(f"dropped empty columns in file {idx + 1}: \n{indices}\n")

    return [reduced_data_files, reduced_renaming_files]


@log(mode=prep.get('mode'))
def renaming_columns(files, renaming_files):
    """ renames columns of pandas DataFrame
    :param files:
    :param renaming_files:
    :return:
    """
    renamed_files = []

    for idx, file in enumerate(files):
        renaming_file = renaming_files[idx]
        col_names = renaming_file['working_title'].to_list()
        dictionary = dict(zip(file.columns.to_list(), col_names))
        file.columns = col_names  # rename data file with 'working_title'
        renamed_files.append(file)
        # logging.info(f"renaming file {idx + 1} (old name / new name): \n\n"
        #              f"{pformat(dictionary)}\n")
    return renamed_files


@log(mode=prep.get('mode'))
def remove_columns(files, col):
    """ removes redundant or unimportant columns from DataFrames
    :param files: list of pandas DataFrames
    :param col: list of colnames which should be removed in file.
    It has to be same number of lists in columns list as file exists.
    :return: list of processed files
    """
    for idx, file in enumerate(files):
        if len(col[idx]) != 0:
            intersect = [item for item in col[idx] if item in file.columns.to_list()]
            diff = [item for item in col[idx] if item not in file.columns.to_list()]
            if len(intersect) != 0:
                file.drop(intersect, axis=1, inplace=True)
                # logging.info(f"file {idx + 1}: successfully dropped columns! ({len(intersect)})\n\n "
                #              f"{pformat(intersect)}\n")
            if len(diff) > 0:
                pass
                # logging.warning(f"failed to drop columns: \n{pformat(diff)}\n")
    return files


@log(mode=prep.get('mode'))
def merging_files(files):
    # merge files by ID -> column 1
    file = reduce(lambda left, right: pd.merge(left, right, on='ID'), files)
    file_before_preprocessing = file.copy()
    # logging.info(f"successfully merged all {len(files)} files!\n")
    # logging.info(f"shape (rows/cols) of new file: {file.shape}\n")
    return file, file_before_preprocessing


@log(mode=prep.get('mode'))
def value_replacement(file):
    # logging.info(f"try value_replacement on columns: \n{pformat(prep['value_replacement'])}\n")
    for item in prep['value_replacement']:
        existing_column = set([True for i in file.columns.to_list() if item['column'] in i])
        if existing_column:
            file = file.apply(replace_cell_content, header=item['column'],
                              pattern=item['pattern'], value=item['value'], axis=0)
            # logging.info(f"value replacement on column {item['column']} successful")
        else:
            pass
            # logging.warning(f"no column '{item['column']}' found!")
    # logging.info(f"'value_replacement' successfully finished\n\n")
    return file


@log(mode=prep.get('mode'))
def force_to_numeric(file):
    # to_numeric (columns with text -> text will be removed in a first step)
    # invalid inputs (strings) will be replaced with nan
    # logging.info(f"try force_to_numeric on columns: \n{pformat(prep['forced_numeric'])}\n")

    for item in prep['forced_numeric']:
        if item in file.columns.to_list():
            try:
                file[item] = file[item].apply(pd.to_numeric, errors='coerce')
            except:
                pass
                # logging.warning(f"failed to convert column {item} to numeric")
            # logging.info(f"force numeric on column {item} successful")
        else:
            pass
            # logging.warning(f"column {item} is not in data! -> force_to_numeric failed!")

    # logging.info(f"'force_to_numeric' successfully finished\n\n")
    return file


@log(mode=prep.get('mode'))
def replace_to_nan(file):
    # logging.info(f"try replace with nan: \n{pformat(prep['replace_with_nan'])}\n")
    for col in file:
        no_occurences = file[col].value_counts()
        occurences = no_occurences.index.to_list()
        intersect = [value for value in occurences if value in prep['replace_with_nan']]
        if len(intersect) > 0:
            for item in intersect:
                file[col] = file[col].replace(item, np.nan)
                # logging.info(f"replaced {no_occurences[item]} '{item}' in column '{col}' with np.nan")

    # logging.info(f"'replace_to_nan' successfully finished\n\n")
    return file


@log(mode=prep.get('mode'))
def repl_text_with_initial_number(file):
    # logging.info(f"try replace initial number with text (exp: '64,0 hemolyze' -> 64.0 or 'clut' -> np.nan): "
    #              f"\n{format(prep['repl_text_with_number'])}\n")

    for item in prep['repl_text_with_number']:
        if item in file.columns.to_list():
            # initially solved with 'apply' but replaced with 'for-loop' due better logging possibilities (each case)
            # file[item] = file[item].apply(replace_text_with_number, args=[logging, item])
            counter = 0
            for idx, cell in enumerate(file[item]):
                cell_temp = cell
                if pd.isna(cell) is False:
                    try:
                        cell = float(cell)
                    except:
                        cell = cell.split()[0]
                        if any(char.isdigit() for char in cell):  # checks if there are numbers in string
                            cell = float(cell.replace(',', '.'))
                        else:
                            cell = np.nan  # if it's only text information set value to 'nan'

                        counter += 1
                        file.loc[idx, item] = cell  # replaces cell content

            # logging.info(f"{counter} cases found in column '{item}'")
        else:
            pass
            # logging.warning(f"column {item} not found!")

    # logging.info(f"number replacement successfully finished\n\n")
    return file


def convert_column_type(file):
    file = convert_ordered_categorical(file)
    file = convert_categorical(file)
    file = convert_categorical_binaries(file)
    file = convert_date(file)
    file = convert_object(file)
    file = convert_numerical(file)
    return file


@log(mode=prep.get('mode'))
def convert_categorical(file):
    categorical = prep['convert_categorical']
    # logging.info(f"try to convert to categorical: \n{format(categorical)}\n")
    file[categorical] = file[categorical].apply(lambda x: x.astype('category'))
    # logging.info(f"'convert_categorical' successfully finished!\n")
    return file


@log(mode=prep.get('mode'))
def convert_categorical_binaries(file):
    categorical = prep['convert_categorical_binaries']
    # logging.info(f"try to convert to categorical binary ['0','1']: \n{format(categorical)}\n")
    file[categorical] = file[categorical].astype(str)
    file[categorical] = file[categorical].replace({'0.0': '0', '1.0': '1'})
    cat_type = CategoricalDtype(categories=["0", "1"], ordered=False)
    file[categorical] = file[categorical].apply(
        lambda x: x.astype(cat_type))  # convert to string, then to binary category
    # logging.info(f"'convert_categorical_binaries' successfully finished!\n")
    return file


@log(mode=prep.get('mode'))
def convert_date(file):
    date = prep['convert_date']
    # logging.info(f"try to convert to date: \n{format(date)}\n")
    file[date] = file[date].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
    # logging.info(f"'convert_date' successfully finished!\n")
    return file


@log(mode=prep.get('mode'))
def convert_numerical(file):
    categorical = prep['convert_categorical']
    categorical_binary = prep['convert_categorical_binaries']
    date = prep['convert_date']
    object = [col for col in file.columns if 'mews_' in col]
    numerical = [item for item in file.columns.to_list() if
                 item not in categorical and item not in date and item not in object and item not in categorical_binary]  # numerical columns (all the rest)
    # logging.info(f"try to convert to numerical: \n{format(numerical)}\n")
    file[numerical] = file[numerical].apply(pd.to_numeric,
                                            errors='ignore')  # If 'ignore', then invalid parsing will return the input
    # logging.info(f"'convert_numerical' successfully finished!\n")
    return file


@log(mode=prep.get('mode'))
def convert_object(file):
    object = [col for col in file.columns if 'mews_' in col]
    # logging.info(f"try to convert to type object: \n{format(object)}\n")
    file[object] = file[object].apply(lambda x: x.astype('object'))
    # logging.info(f"'convert_object' successfully finished!\n")
    return file


@log(mode=prep.get('mode'))
def convert_ordered_categorical(file):
    # logging.info(f"try 'convert_ordered_categorical': \n{format(prep['convert_ordered_categorical'])}\n ")

    for item in prep['convert_ordered_categorical']:
        if item[0] in file.columns.to_list():
            cat_type = CategoricalDtype(categories=item[1], ordered=True)
            try:
                file[item[0]] = file[item[0]].astype(cat_type)
                # logging.info(f"successfully converted column {item[0]}")
            except:
                pass
                # logging.warning(f"failed to convert column {item[0]}")
        else:
            pass
            # logging.warning(f"column {item} not found!")

    # logging.info(f"'convert_ordered_categorical' successfully finished\n\n")
    return file


def build_quotient_ts(file):
    for item in prep['build_quotient_ts']:
        nom = [i for i in list(file.columns) if i.startswith(item[0]+'_')]
        denom = [i for i in list(file.columns) if i.startswith(item[1]+'_')]

        # take shorter sequence and make longer sequence that short
        if len(nom) >= len(denom):
            nom = nom[0:len(denom)]
        else:
            denom = denom[0:len(nom)]

        for idx, _ in enumerate(nom):
            name = item[0] + 'Q' + item[1] + '_' + str(idx+1)
            file[name] = file.loc[:,nom[idx]] / file.loc[:,denom[idx]]

    return file


def remove_ts(file):

    for item in prep['remove_ts']:
        cols = [i for i in list(file.columns) if i.startswith(item)]
        file.drop(cols, axis=1, inplace=True)
    return file


if __name__ == '__main__':
    run()

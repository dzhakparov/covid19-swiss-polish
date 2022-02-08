from src.helpers import log
from config import general
import pandas as pd



@log()
def read_in_files(filenames):
    """
    read-in file-names from a list of strings and returns a list of lies of same length
    :param filenames: list of filenames as string
    :return: list of files
    """
    path = general['input_path']
    decimal = ['.', ',', ',', ',']  # TODO: add config

    files = []
    for idx, item in enumerate(filenames):
        if decimal is not None:
            dec = decimal[idx]
        else:
            dec = None

        df = pd.read_csv(f'{path}{item}.csv', decimal=dec)
        df.rename(columns={df.columns[0]: "ID"}, inplace=True)
        # file = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # remove columns with header 'unnamed'
        file = df.copy()
        files.append(file)

    return files
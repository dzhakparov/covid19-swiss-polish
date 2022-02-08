from src.helpers import get_wd
from config import general
import os
import shutil


def setup_system():
    """ build desired folders to store resulting output / figures
    :return: None
    """
    sub_pathes = ['sax', 'log', 'fig', 'fig/boxplot', 'fig/survivalcurve']

    # build main directory
    path = os.path.join(get_wd(), general['output_path'])
    if not os.path.isdir(path):
        os.mkdir(path)

    # build or clear sub_directories
    for sub_path in sub_pathes:
        path = os.path.join(get_wd(), general['output_path'], sub_path)
        if not os.path.isdir(path):
            os.mkdir(path)
        else:
            _remove_files_in_folder(path)


def _remove_files_in_folder(folder):
    """ clears all data from folder given
    :param folder: path to folder which should be cleared (string)
    :return: None
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
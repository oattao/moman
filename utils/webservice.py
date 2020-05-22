import os
import pathlib
import pandas as pd
from configs.image import DATA_PATH, IMAGE_FILE_EXTENSIONS

def count_image(folder):
    datadir = pathlib.Path(folder)
    num_images = 0
    for ext in IMAGE_FILE_EXTENSIONS:
        num_images += len(list(datadir.glob('*.{}'.format(ext))))
    return num_images


def show_tabula(folder_list):
    # Get class
    data = {}
    for folder in folder_list:
        folder_path = os.path.join(DATA_PATH, folder)
        class_list = [class_name for class_name in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, class_name))]
        if len(class_list) > 0:
            subdict = {}
            for class_name in class_list:
                num_images = count_image(os.path.join(folder_path, class_name))
                subdict[class_name] = num_images
            data[folder] = subdict
    return data
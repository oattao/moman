import os
import pathlib
import pandas as pd
from configs.image import DATA_PATH, IMAGE_FILE_EXTENSIONS
from configs.server import MODEL_LOG

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

def get_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

def getsize_h5model(folder):
    return os.path.getsize(folder)

def log_write(log_path, log_content):
    log_frame = dict(zip(MODEL_LOG, log_content))
    df_new = pd.DataFrame(log_frame, index=[0])
    # Check if file exist
    if os.path.isfile(log_path):
        df_old = pd.read_csv(log_path)
        df_new = pd.concat((df_old, df_new), axis=0, ignore_index=True)
    df_new.to_csv(log_path, index=None)    

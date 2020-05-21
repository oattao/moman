from flask import Blueprint, Flask, request, render_template

import os
import sys
import shutil
import pathlib
import pandas as pd
from configs.image import DATA_PATH, IMAGE_FILE_EXTENSIONS

def count_image(folder):
    datadir = pathlib.Path(folder)
    num_images = 0
    for ext in IMAGE_FILE_EXTENSIONS:
        num_images += len(list(datadir.glob('*.{}'.format(ext))))
    return num_images

manage_data = Blueprint('manage_data', __name__)

@manage_data.route('/manage_data')
def showpage():
    # get subfolder of data path
    cols = ['Folder', 'Class', 'Number of image']

    folder_list = [name for name in os.listdir(DATA_PATH)\
              if os.path.isdir(os.path.join(DATA_PATH, name))] 
    if len(folder_list) > 0:
        folder_frame = pd.DataFrame(columns=cols)
        # Get class
        for folder in folder_list:
            folder_path = os.path.join(DATA_PATH, folder)
            class_list = [class_name for class_name in os.listdir(folder_path)\
                          if os.path.isdir(os.path.join(folder_path, class_name))]
            if len(class_list) > 0:
                for i, class_name in enumerate(class_list):
                    num_images = count_image(os.path.join(folder_path, class_name))
                    if i == 0:
                        row_data = dict(zip(cols, [folder, class_name, num_images]))
                    else:
                        row_data = dict(zip(cols, ['', class_name, num_images]))
                    folder_frame = folder_frame.append(pd.DataFrame(row_data, index=[0]), 
                                                       ignore_index=True)

        data_cols = [folder_frame[col].values for col in cols]
        data = zip(*data_cols)
        return render_template('image_page.html', cols=cols, data=data)

    return render_template('image_page.html', display='none')



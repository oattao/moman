from flask import Blueprint, Flask, request, render_template

import os
import sys
import shutil
import pathlib
import pandas as pd
from utils.webservice import show_tabula
from configs.image import DATA_PATH, IMAGE_FILE_EXTENSIONS

manage_data = Blueprint('manage_data', __name__)

@manage_data.route('/manage_data')
def showpage():
    # get subfolder of data path
    folder_list = [name for name in os.listdir(DATA_PATH)\
              if os.path.isdir(os.path.join(DATA_PATH, name))] 
    if len(folder_list) > 0:
        data = show_tabula(folder_list)
        return render_template('image_page.html', data=data)
    return render_template('image_page.html', display='none')



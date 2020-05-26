from flask import Blueprint, Flask, request, render_template, jsonify
import os
import signal
import time
import platform
import pickle
import shutil
import pandas as pd
from utils.model import train_model
from utils.webservice import show_tabula
from configs.image import DATA_PATH
from configs.server import MODEL_PATH, LOG_FILE, FLAG, HIST, NEED_CONFIRM
import pdb

train = Blueprint('train', __name__)

@train.route('/stoptraining', methods=['POST'])
def stoptraining():
    with open(os.path.join(MODEL_PATH, FLAG), 'rb') as f:
        flag = pickle.load(f)
        pid = flag['pid']
        model_id = flag['model_id']
    os.kill(pid, signal.SIGTERM)
    os.remove(os.path.join(MODEL_PATH, FLAG))
    hist_file = os.path.join(MODEL_PATH, HIST)
    if os.path.exists(hist_file):
        os.remove(os.path.join(MODEL_PATH, HIST))
    if os.path.exists(os.path.join(MODEL_PATH, model_id)):
        shutil.rmtree(os.path.join(MODEL_PATH, model_id))

    folder = [name for name in os.listdir(DATA_PATH)\
                  if os.path.isdir(os.path.join(DATA_PATH, name))] 
    if len(folder) > 0:
        data = show_tabula(folder)
    else:
        data = None
    return render_template('train_page.html', data=data, train_status=0)

@train.route('/changemodel', methods=['POST'])
def changemodel():
    # is_busy = False
    model_name = request.json['model_name']
    command = request.json['command']
    df = pd.read_csv(os.path.join(MODEL_PATH, LOG_FILE))
    idx = df[df['ID'] == model_name].index
    if command == 'discard':
        model_path = os.path.join(MODEL_PATH, model_name)
        # delete model folder
        shutil.rmtree(model_path)
        df.drop(idx, axis=0, inplace=True)
    if command == 'save':
        df.at[idx, '_is_confirmed'] = True

    df.to_csv(os.path.join(MODEL_PATH, LOG_FILE), index=None)

    # delete temp history
    os.remove(os.path.join(MODEL_PATH, HIST))
    os.remove(os.path.join(MODEL_PATH, NEED_CONFIRM))
    return jsonify({'confirm': 'yes'})

@train.route('/', methods=['GET'])
def showpage():
    # get the image folder list
    folder = [name for name in os.listdir(DATA_PATH)\
              if os.path.isdir(os.path.join(DATA_PATH, name))] 
    if len(folder) > 0:
        data = show_tabula(folder)
    else:
        data = None

    # set training status
    train_status = 0   # no training, no need confirm

    # check if server is busy on training task
    if os.path.exists(os.path.join(MODEL_PATH, FLAG)):
        train_status = 1 # is busy training model
    elif os.path.exists(os.path.join(MODEL_PATH, NEED_CONFIRM)):
        train_status = 2 # training done, need confirm

    return render_template('train_page.html',
                            train_status=train_status,
                            data=data)


@train.route('/', methods=['POST'])
def trainmodel():
    folder = [name for name in os.listdir(DATA_PATH)\
                  if os.path.isdir(os.path.join(DATA_PATH, name))] 
    if len(folder) > 0:
        data = show_tabula(folder)
    else:
        data = None

    # Handle POST request
    parameters = request.form
    learning_rate = float(parameters.get('lrInputName'))
    image_folder = parameters.get('folder')
    model_name = parameters.get('model')
    num_epochs = int(parameters.get('epoch'))

    if platform.system() == 'Linux':
        os.system('python3 train_process.py {} {} {} {}'.format(model_name,
            image_folder, num_epochs, learning_rate))
    else:
        os.system("start cmd.exe /c python train_process.py {} {} {} {}".format(
            model_name,image_folder, num_epochs, learning_rate))

    return render_template('train_page.html', data=data, train_status=1)
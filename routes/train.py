from flask import Blueprint, request, render_template, jsonify
import os
import json
import requests
import signal
import time
import platform
import pickle
import shutil
import pandas as pd
from utils.webservice import show_tabula
from configs.image import DATA_PATH
from configs.server import MODEL_PATH, LOG_FILE, FLAG, HIST, NEED_CONFIRM
from configs.server import API_HOST, API_PORT
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
        # shutil.rmtree(os.path.join(MODEL_PATH, model_id))
        os.remove(os.path.join(MODEL_PATH, '{}.h5'.format(model_id)))

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
        model_path = os.path.join(MODEL_PATH, '{}.h5'.format(model_name))
        # delete model folder
        # shutil.rmtree(model_path)
        os.remove(model_path)
        df.drop(idx, axis=0, inplace=True)
    if command == 'save':
        df.at[idx, '_is_confirmed'] = True
        # load new model to the api
        data = json.dumps({"signature_name": "serving_default", "model_name": model_name})
        headers = {"content-type": "application/json"}
        # Select Flask RESTapi serving
        json_response = requests.post('http://{}:{}/add/{}'.format(API_HOST, API_PORT, model_name),
                                      data=data, headers= headers)   


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
    # Fake
    import time
    time.sleep(2)

    return render_template('train_page.html', data=data, train_status=1)
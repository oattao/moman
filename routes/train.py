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
from configs.server import MODEL_PATH, LOG_FILE, FLAG, HIST
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
    if os.path.exists(os.path.join(MODEL_PATH, model_id)):
        shutil.rmtree(os.path.join(MODEL_PATH, model_id))

    folder = [name for name in os.listdir(DATA_PATH)\
                  if os.path.isdir(os.path.join(DATA_PATH, name))] 
    if len(folder) > 0:
        data = show_tabula(folder)
    else:
        data = None
    return render_template('train_page.html', data=data, stop=True, is_busy=False)


@train.route('/changemodel', methods=['POST'])
def changemodel():
    # is_busy = False
    model_name = request.json['model_name']
    command = request.json['command']
    df = pd.read_csv(os.path.join(MODEL_PATH, LOG_FILE))
    idx = df[df['ID'] == model_name].index
    # pdb.set_trace()
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

    # check if server is busy on training task
    if os.path.exists(os.path.join(MODEL_PATH, FLAG)):
        is_busy = True
    else:
        is_busy = False

    # check if any model waiting for confirm
    if os.path.exists(os.path.join(MODEL_PATH, HIST)):
        with open(os.path.join(os.path.join(MODEL_PATH, HIST)), 'rb') as f:
            history = pickle.load(f)
        accuracy = [100 * n for n in history['accuracy']]
        val_accuracy = [100 * n for n in history['val_accuracy']]
        loss = history['loss']
        val_loss = history['val_loss']

        log = os.path.join(MODEL_PATH, LOG_FILE)
        logdata = pd.read_csv(log).iloc[-1]
        model_name = logdata['ID']
        acc = logdata['Accuracy']
        return render_template('train_page.html',
                                is_busy=is_busy,
                                model_name=model_name, acc=acc, 
                                val_loss=val_loss,
                                val_accuracy=val_accuracy,
                                loss=loss,
                                accuracy=accuracy,
                                data=data)
    else:
        return render_template('train_page.html',
                               is_busy=is_busy,
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
        os.system('python train_process.py {} {} {} {}'.format(model_name,
            image_folder, num_epochs, learning_rate))
    else:
        os.system("start cmd.exe /c python train_process.py {} {} {} {}".format(
            model_name,image_folder, num_epochs, learning_rate))

    return render_template('train_page.html', data=data, is_busy=True)

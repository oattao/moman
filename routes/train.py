from flask import Blueprint, Flask, request, render_template, jsonify
from sys import executable
from subprocess import Popen, CREATE_NEW_CONSOLE
import os
import pickle
import shutil
import pandas as pd
from utils.model import train_model
from utils.webservice import show_tabula
from configs.image import DATA_PATH
from configs.server import MODEL_PATH, LOG_FILE
import pdb

train = Blueprint('train', __name__)

@train.route('/discard', methods=['POST'])
def discard():
    model_name = request.json['model_name']
    model_path = os.path.join(MODEL_PATH, model_name)
    # delete model folder
    shutil.rmtree(model_path)

    # delete in the csv file
    df = pd.read_csv(os.path.join(MODEL_PATH, LOG_FILE))
    del_idx = df[df['ID'] == model_name].index
    df.drop(del_idx, axis=0, inplace=True)
    df.to_csv(os.path.join(MODEL_PATH, LOG_FILE), index=None)

    return jsonify({'confirm_discard': 'yes'})

@train.route('/', methods=['GET', 'POST'])
def showpage():
    # check if any model is training?
    folder = [name for name in os.listdir(DATA_PATH)\
                  if os.path.isdir(os.path.join(DATA_PATH, name))] 
    if len(folder) > 0:
        data = show_tabula(folder)
    else:
        data = None
    
    if request.method == 'GET':
        tmp = '_tmp'
        if os.path.exists('_tmp'):
            if os.path.exists(os.path.join(tmp, 'tmp_hist.pickle')):
                print('Model finish training.')
                # Show up the history
                with open(os.path.join(tmp, 'tmp_hist.pickle'), 'rb') as f:
                    history = pickle.load(f)
                with open(os.path.join(tmp, 'tmp_log.pickle'), 'rb') as f:
                    log = pickle.load(f)

                accuracy = [100 * n for n in history['accuracy']]
                val_accuracy = [100 * n for n in history['val_accuracy']]
                loss = history['loss']
                val_loss = history['val_loss']

                model_name = log['ID']
                acc = log['Accuracy']
                return render_template('train_page.html',
                                        model_name=model_name, acc=acc, 
                                        val_loss=val_loss,
                                        val_accuracy=val_accuracy,
                                        loss=loss,
                                        accuracy=accuracy,
                                        data=data)

            return render_template('train_page.html', is_busy=True, data=data)
        return render_template('train_page.html', data=data)

    # Handle POST request
    parameters = request.form
    learning_rate = float(parameters.get('lrInputName'))
    image_folder = parameters.get('folder')
    model_name = parameters.get('model')
    num_epochs = int(parameters.get('epoch'))

    # Somthing new here
    # pid = Popen([executable, 'train_process.py {} {} {} {}'.format(model_name,
    #     image_folder, num_epochs, learning_rate)], creationflags=CREATE_NEW_CONSOLE).pid
    # print('Model is being trained as pid {}'.format(pid))

    os.system("start cmd.exe /c python train_process.py {} {} {} {}".format(
        model_name,image_folder, num_epochs, learning_rate))

    return render_template('train_page.html', data=data)
    # End hope somthing new here

    """
    acc, history, trained_model = train_model(model_name=model_name, image_folder=image_folder, 
                                              num_epochs=num_epochs, learning_rate=learning_rate)    

    # round values
    accuracy = [100 * n for n in history['accuracy']]
    val_accuracy = [100 * n for n in history['val_accuracy']]
    loss = history['loss']
    val_loss = history['val_loss']

    return render_template('train_page.html', folder=folder,
                            model_name=trained_model, acc=acc, 
                            val_loss=val_loss,
                            val_accuracy=val_accuracy,
                            loss=loss,
                            accuracy=accuracy,
                            data=data)
    """

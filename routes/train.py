from flask import Blueprint, Flask, request, render_template, jsonify
import os
import shutil
import pandas as pd
from utils.model import train_model
from configs.image import DATA_PATH
from configs.server import MODEL_PATH, LOG_FILE

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

@train.route('/train', methods=['GET', 'POST'])
def showpage():
    if request.method == 'GET':
        folder = [name for name in os.listdir(DATA_PATH)\
         if os.path.isdir(os.path.join(DATA_PATH, name))] 
        if len(folder) == 0:
            data_status = 3
        else:
            data_status = 0

        return render_template('train_page.html', folder=folder, data_status=data_status)

    # Handle POST request
    parameters = request.form
    learning_rate = float(parameters.get('lrInputName'))
    image_folder = parameters.get('folder')
    model_name = parameters.get('model')
    num_epochs = int(parameters.get('epoch'))

    acc, history, trained_model = train_model(model_name=model_name, image_folder=image_folder, 
                                              num_epochs=num_epochs, learning_rate=learning_rate)

    # get subfolder of data path
    folder = [name for name in os.listdir(DATA_PATH)\
         if os.path.isdir(os.path.join(DATA_PATH, name))] 
    if len(folder) == 0:
        data_status = 3
    else:
        data_status = 0       

    # round values
    accuracy = [100 * n for n in history['accuracy']]
    val_accuracy = [100 * n for n in history['val_accuracy']]
    loss = history['loss']
    val_loss = history['val_loss']

    return render_template('train_page.html', folder=folder, data_status=data_status,
                            model_name=trained_model, acc=acc, 
                            val_loss=val_loss,
                            val_accuracy=val_accuracy,
                            loss=loss,
                            accuracy=accuracy)


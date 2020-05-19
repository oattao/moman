import os
import shutil
import numpy as np
from flask import Flask, render_template, request, jsonify

from utils.model import train_model
from configs.image import DATA_PATH
from configs.server import MODEL_PATH
import pdb

app = Flask(__name__)

@app.route('/')
def manager():
    # get subfolder of data path
    folder = [name for name in os.listdir(DATA_PATH)\
         if os.path.isdir(os.path.join(DATA_PATH, name))] 
    if len(folder) == 0:
        data_status = 3
    else:
        data_status = 0

    return render_template('manager.html', folder=folder, data_status=data_status)

@app.route('/train', methods=['POST'])
def listen():
    # parameters = request.json
    # learning_rate = parameters['lr']
    # image_folder = parameters['image_folder']
    # model_name = parameters['model_name']
    # num_epochs = int(parameters['num_epochs'])
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

    return render_template('manager.html', folder=folder, data_status=data_status,
                            model_name=trained_model, acc=acc, 
                            val_loss=val_loss,
                            val_accuracy=val_accuracy,
                            loss=loss,
                            accuracy=accuracy)

@app.route('/discard', methods=['POST'])
def discard():
    model_name = request.json['model_name']
    model_path = os.path.join(MODEL_PATH, model_name)
    print(model_path)
    shutil.rmtree(model_path)
    return jsonify({'confirm_discard': 'yes'})

if __name__ == "__main__":
    app.run(debug=True, port=8080)


import os
import shutil
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for, flash

from utils.model import train_model, load_model
from utils.image import allowed_file, load_image
from configs.image import DATA_PATH, CLASSES
from configs.server import MODEL_PATH, LOCAL_HOST, PUBLIC_HOST, WEB_SERVER_PORT, UPLOAD_FOLDER
import pdb

app = Flask(__name__)

model_list = None

@app.route('/index')
def index():
    # get subfolder of data path
    folder = [name for name in os.listdir(DATA_PATH)\
         if os.path.isdir(os.path.join(DATA_PATH, name))] 
    if len(folder) == 0:
        data_status = 3
    else:
        data_status = 0

    return render_template('train_page.html', folder=folder, data_status=data_status)

@app.route('/manage_models')
def manage_models():
     # load trained model
    models = [name for name in os.listdir(MODEL_PATH)\
              if os.path.isdir(os.path.join(MODEL_PATH, name))] 
    num_models = len(models)
    return render_template('model_page.html', models=models, num_models=num_models)

@app.route('/predict', methods=['GET', 'POST'])
def predict_route():
    global model_list
    # load trained model
    models = [name for name in os.listdir(MODEL_PATH)\
              if os.path.isdir(os.path.join(MODEL_PATH, name))] 
    num_models = len(models)

    if request.method == 'GET':
        if num_models > 0:
            model_list = load_model(models)
        return render_template('predict_page.html', models=models, num_models=num_models)
    if request.method == 'POST':
        # handle the image
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            image = load_image(filepath, to_4d=True, normalized=True)

            # get model name selected by client
            model_name = request.form.get('model')
            prediction = model_list[model_name].predict(image)[0]
            prob = [round(p*100, 2) for p in prediction]
            
             # Get car model
            p0 = '{}: {} %'.format(CLASSES[0], prob[0])
            p1 = '{}: {} %'.format(CLASSES[1], prob[1])
            p2 = '{}: {} %'.format(CLASSES[2], prob[2])
            p3 = '{}: {} %'.format(CLASSES[3], prob[3])

            return render_template('predict_page.html', models=models, num_models=num_models,
                                   p0=p0, p1=p1, p2=p2, p3=p3,
                                   model_name=model_name,
                                   filepath=filepath)


@app.route('/image', methods=['GET', 'POST'])
def image_route():
    # get subfolder of data path
    folder = [name for name in os.listdir(DATA_PATH)\
         if os.path.isdir(os.path.join(DATA_PATH, name))] 
    if len(folder) == 0:
        data_status = 3
    else:
        data_status = 0
    return render_template('image_page.html', folder=folder, data_status=data_status)


@app.route('/train', methods=['GET', 'POST'])
def train_route():
    parameters = request.form
    learning_rate = float(parameters.get('lrInputName'))
    image_folder = parameters.get('folder')
    model_name = parameters.get('model')
    num_epochs = int(parameters.get('epoch'))

    try:
        acc, history, trained_model = train_model(model_name=model_name, image_folder=image_folder, 
                                                    num_epochs=num_epochs, learning_rate=learning_rate)
        training_success = True
    except:
        training_success = False

    # get subfolder of data path
    folder = [name for name in os.listdir(DATA_PATH)\
         if os.path.isdir(os.path.join(DATA_PATH, name))] 
    if len(folder) == 0:
        data_status = 3
    else:
        data_status = 0

    if training_success == False:
        return render_template('train_page.html', folder=folder, data_status=data_status,
                                training_fail = True)        

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

@app.route('/discard', methods=['POST'])
def discard():
    model_name = request.json['model_name']
    model_path = os.path.join(MODEL_PATH, model_name)
    print(model_path)
    shutil.rmtree(model_path)
    return jsonify({'confirm_discard': 'yes'})

if __name__ == "__main__":
    app.run(debug=True, host=LOCAL_HOST, port=WEB_SERVER_PORT)


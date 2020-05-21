from flask import Blueprint, Flask, request, render_template
import os
from utils.model import load_model
from utils.image import allowed_file, load_image
from configs.image import CLASSES
from configs.server import MODEL_PATH, UPLOAD_FOLDER

predict = Blueprint('predict', __name__)

model_list = None
import pdb

@predict.route('/predict', methods=['GET', 'POST'])
def showpage():
    global model_list
    # load trained model
    models = [name for name in os.listdir(MODEL_PATH)\
              if os.path.isdir(os.path.join(MODEL_PATH, name))] 
    num_models = len(models)

    if request.method == 'GET':
        if num_models > 0:
            model_list = load_model(models)
        return render_template('predict_page.html', models=models, num_models=num_models)

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
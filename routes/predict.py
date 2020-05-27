from flask import Blueprint, request, render_template, flash
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

import os
import pandas as pd
from utils.model import load_model
from utils.image import allowed_file, load_image
from configs.image import CLASSES
from configs.server import MODEL_PATH, UPLOAD_FOLDER, LOG_FILE

predict = Blueprint('predict', __name__)

# https://gist.github.com/greyli/ca74d71f13c52d089a8da8d2b758d519
class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image Only!'), FileRequired('Choose a file!')])
    submit = SubmitField('Upload')

model_list = None
import pdb

@predict.route('/predict', methods=['GET', 'POST'])
def showpage():
    global model_list

    form = UploadForm()
    # load trained model
    # check if model database is exist
    if not os.path.exists(os.path.join(MODEL_PATH, LOG_FILE)):
        return render_template('model_page.html', display='none')
    # read database
    df = pd.read_csv(os.path.join(MODEL_PATH, LOG_FILE))
    df = df[df['_is_confirmed'] == True]
    num_models = len(df)

    if num_models>0:
        cols = list(df.columns)
        cols.pop(-1)
        data_cols = [df[col].values for col in cols]
        data = zip(*data_cols)

    if request.method == 'GET':
        if num_models == 0:
            return render_template('predict_page.html', display='none')
        else:
            names = df['ID'].values
            model_list = load_model(names)
            return render_template('predict_page.html', cols=cols, data=data)

    # handle the image
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No file selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # try fix the filename
        # filename = 'uploaded.jpg'
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

        return render_template('predict_page.html',
                                p0=p0, p1=p1, p2=p2, p3=p3,
                                model_name=model_name,
                                filepath=filepath,
                                cols=cols, data=data)
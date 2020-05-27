import os 
import sys

import tensorflow as tf
from flask import Flask, jsonify, make_response, request, abort

from utils.model import load_model
from configs.server import MODEL_PATH, LOG_FILE, API_HOST, API_PORT

model_list = load_model(os.path.join(MODEL_PATH, LOG_FILE))

app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/', methods=['GET'])
def info():
    return "Predict item quality by input images."

@app.route('/<model_name>', methods=['GET'])
def detailinfo(model_name):
    if model_name not in model_list.keys():
        return "Model '{}'' is currently not supported.".format(model_name)
    return "Model '{}' is ready to serve".format(model_name)

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    if not request.json or not 'instances' in request.json:
        abort(400)
    image = request.json['instances']
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    predictor = model_list[model_name]
    prob = predictor.predict(image).tolist()
    print('{} finish predicting job.'.format(model_name))
    return jsonify({'predictions': prob})

@app.route('/add/<model_name>', methods=['POST'])
def add(model_name):
    model_list[model_name] = tf.keras.models.load_model(
                os.path.join(MODEL_PATH, '{}.h5'.format(model_name)))
    print('Model {} added'.format(model_name))
    return jsonify({model_name: 'added'})

@app.route('/delete/<model_name>', methods=['POST'])
def delete(model_name):
    model_list.pop(model_name, None)
    print('Model {} deleted'.format(model_name))
    return jsonify({model_name: 'deleted'})

if __name__ == "__main__":
    print('API is served at port {}'.format(API_PORT))
    app.run(debug=True, host=API_HOST, port=API_PORT)





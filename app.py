import os
import pickle
import json
import argparse

from flask import Flask, request
from flask_socketio import SocketIO, emit
from gevent.pywsgi import WSGIServer

from configs.server import HIST, MODEL_PATH, FLAG

from routes.manage_data import manage_data
from routes.manage_model import manage_model
from routes.predict import predict
from routes.train import train
from routes.monitor import monitor
from routes.error import error

parser = argparse.ArgumentParser()
parser.add_argument("HOST", type=str, help="web host")
parser.add_argument("PORT", type=str, help="web port")
args = parser.parse_args()

app = Flask(__name__)
app.register_blueprint(manage_data)
app.register_blueprint(manage_model)
app.register_blueprint(predict)
app.register_blueprint(train)
app.register_blueprint(monitor)
app.register_blueprint(error)

# app.config.update(DEBUG=True)
app.config.update(DEBUG=True, SERVER_NAME ="{}:{}".format(args.HOST, args.PORT))

socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)

@app.route("/monitortraining", methods=['POST'])
def publish():
    payload = request.form.get('data')
    try:
        data = json.loads(payload)
        num_epochs = data['epoch'] + 1
        loss = round(data['loss'], 2)
        val_loss = round(data['val_loss'], 2)
        accuracy = round(data['accuracy'] * 100, 2)
        val_accuracy = round(data['val_accuracy'] * 100, 2)
        data = {'loss': loss, 'val_loss': val_loss, 
                'accuracy': accuracy, 'val_accuracy': val_accuracy}

        socketio.emit('newdata', {'data': data}, namespace="/monitortraining")
        

        # Save history to HIST file
        hist_file = os.path.join(MODEL_PATH, HIST)
        if os.path.exists(hist_file):
            with open(hist_file, 'rb') as f:
                history = pickle.load(f)
                history['loss'].append(loss)
                history['val_loss'].append(val_loss)
                history['accuracy'].append(accuracy)
                history['val_accuracy'].append(val_accuracy)
        else:
            history = {'loss': [loss], 'val_loss': [val_loss], 
                       'accuracy': [accuracy], 'val_accuracy': [val_accuracy]}

        with open(hist_file, 'wb') as f:
            pickle.dump(history, f)

        # check if finish epoch
        with open(os.path.join(MODEL_PATH, FLAG), 'rb') as f:
            n_epochs = pickle.load(f)['num_epochs']

        if n_epochs == num_epochs:
            socketio.emit('end', {'finish': 'ok'}, namespace="/monitortraining")

    except:
        return {'error': 'invalid payload'}
    return "OK"

# @socketio.on('connect')
# def test_connect():
#     emit('my response', {'data': 'Connected'})    

# @socketio.on('disconnect')
# def test_disconnect():
#     print('Client disconnected')    

if __name__ == "__main__":
    socketio.run(app)
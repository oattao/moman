import os
import pickle
import pandas as pd
from flask import Blueprint, render_template, redirect, url_for
from configs.server import FLAG, MODEL_PATH, LOG_FILE, NEED_CONFIRM, HIST

monitor = Blueprint('monitor', __name__)

@monitor.route('/monitortraining', methods=['GET'])
def monitortraining():
    # read history file
    hist_file = os.path.join(MODEL_PATH, HIST)
    if os.path.exists(hist_file):
        with open(hist_file, 'rb') as f:
            history = pickle.load(f)
            val_loss = history['val_loss']
            loss = history['loss']
            accuracy = history['accuracy']
            val_accuracy = history['val_accuracy']
    else: 
        loss= 5
        val_loss = 5
        accuracy = 0
        val_accuracy = 0

    # check if busy
    if os.path.exists(os.path.join(MODEL_PATH, FLAG)):
        train_status = 1   # busy
        return render_template('monitor_page.html',
                               train_status=train_status,
                               val_loss=val_loss,
                               loss=loss,
                               accuracy=accuracy,
                               val_accuracy=val_accuracy)

    # check if need confirm
    # if os.path.exists(os.path.join(MODEL_PATH, NEED_CONFIRM)):
    else:
        if os.path.exists(os.path.join(MODEL_PATH, LOG_FILE)):
            log = os.path.join(MODEL_PATH, LOG_FILE)
            logdata = pd.read_csv(log).iloc[-1]
            model_name = logdata['ID']
            acc = logdata['Accuracy']
            return render_template('monitor_page.html',
                                    model_name=model_name, acc=acc, 
                                    val_loss=val_loss,
                                    val_accuracy=val_accuracy,
                                    loss=loss,
                                    accuracy=accuracy)
        else:
            return redirect(url_for('train.showpage'))
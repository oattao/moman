import argparse
import pickle
import os
import tensorflow as tf
from datetime import datetime, date
from utils.model import create_model, log_write, get_size
from utils.data import create_dataframe, ImageGenerator
from sklearn.model_selection import train_test_split

from configs.server import MODEL_LOG
from configs.image import DATA_PATH

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="Base model name")
parser.add_argument("image_folder", type=str, help="Image folder")
parser.add_argument("num_epochs", type=int, help="Number of training epochs")
parser.add_argument("learning_rate", type=float, help="Learning rate")

args = parser.parse_args()

model_name = args.model_name
image_folder = args.image_folder
num_epochs = args.num_epochs
learning_rate = args.learning_rate

# get time for log
today = date.today()
time_now = datetime.now().strftime("%H-%M-%S")

# get number of classes (= number of subfolder)
training_path = os.path.join(DATA_PATH, image_folder)
classes = [name for name in os.listdir(training_path)
               if os.path.isdir(os.path.join(training_path, name))]
num_classes = len(classes)

# define model
model = create_model(model_name, num_classes)

# create image dataframe
image_dataframe = create_dataframe(training_path, classes)

# split train - test set
data_split = ['train', 'test']
train_frame, test_frame = train_test_split(
image_dataframe, test_size=0.15, random_state=911)

# image generator
generator = {'train': ImageGenerator(df=train_frame, label_col='Classes', classes=classes),
             'test': ImageGenerator(df=test_frame, label_col='Classes', classes=classes)}

# checkpoint callbacks
tmp = '_tmp'
os.mkdir(tmp) 
folder_name = '{}_{}_{}'.format(model_name, today, time_now)
checkpoint_path = os.path.join(tmp, folder_name)
checkpoint_dir = os.path.dirname(checkpoint_path)
saving_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor='val_accuracy',
                                                     save_best_only=True,
                                                     verbose=1)

history = model.fit(x=generator['train'], validation_data=generator['test'],
                    validation_freq=1, epochs=num_epochs, verbose=1,
                    callbacks=[saving_callback])
history = history.history
accuracy = round(100 * max(history['accuracy']), 2)

# For log
_id = folder_name
_name = model_name
_training_date = today
_training_starttime = time_now
_training_stoptime = datetime.now().strftime("%H-%M-%S")
# get size of model
_size = get_size(checkpoint_path)
m1 = 1024 * 1024
if _size > m1:
    _size = '{} mb'.format(round(_size / m1), 2)
else:
    _size = '{} kb'.format(round(_size / 1024.), 2)

_training_data = image_folder
_accuracy = accuracy
_is_confirmed = False

log_content = [_id, _name, _training_date, _training_starttime,
               _training_stoptime, _training_data, _size, _accuracy, _is_confirmed]

tmp_log = dict(zip(MODEL_LOG, log_content))
tmp_hist = history

with open(os.path.join(tmp, 'tmp_log.pickle'), 'wb') as f:
	pickle.dump(tmp_log, f)
with open(os.path.join(tmp, 'tmp_hist.pickle'), 'wb') as f:
	pickle.dump(tmp_hist, f)





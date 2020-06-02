import argparse
import pickle
import os
import numpy as np
import tensorflow as tf
from datetime import datetime, date
from utils.model import create_model
from utils.webservice import getsize_h5model, log_write
from utils.data import create_dataframe, ImageGenerator, get_sample, config_batch
from sklearn.model_selection import train_test_split

from configs.server import MODEL_PATH, LOG_FILE, FLAG, HIST, NEED_CONFIRM
from configs.image import DATA_PATH
import keras
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

# ID of model
folder_name = '{}_{}_{}'.format(model_name, today, time_now)

# get the pid of current process
pid = os.getpid()

# before training raise a flag
flag = os.path.join(MODEL_PATH, FLAG)
with open(flag, 'wb') as f:
    pickle.dump({'pid': pid, 'model_id': folder_name, 
                 'num_epochs': num_epochs}, f)

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
data_split = ['train', 'val']
train_frame, val_frame = train_test_split(image_dataframe, 
	test_size=0.15, random_state=911)

# image generator
if model_name in ['Simple', 'Small', 'Tiny']:
    batch_size = 32
else:
    batch_size = 8

list_ds = {'train': tf.data.Dataset.list_files(train_frame['Filepath'].values),
		   'val': tf.data.Dataset.list_files(val_frame['Filepath'].values)}
# pdb.set_trace()		  
image_num = {'train': len(train_frame), 'val': len(val_frame)}		
steps_per_epoch = {x: np.ceil(image_num[x] / batch_size) for x in data_split}   

AUTOTUNE = tf.data.experimental.AUTOTUNE
labeled_ds = {x: list_ds[x].map(get_sample, num_parallel_calls=AUTOTUNE) for x in data_split}	
generator = {x: config_batch(ds=labeled_ds[x], batch_size=batch_size) for x in data_split}	   

# generator = {'train': ImageGenerator(df=train_frame, label_col='Classes',
#                                      classes=classes, batch_size=batch_size),
#              'val': ImageGenerator(df=test_frame, label_col='Classes', 
#                                     classes=classes, batch_size=batch_size)}

# checkpoint callback
checkpoint_path = os.path.join(MODEL_PATH, '{}.h5'.format(folder_name))
checkpoint_dir = os.path.dirname(checkpoint_path)
saving_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  monitor='val_accuracy',
                                                  save_best_only=True,
                                                  verbose=1)
# tensorboard callback
tfboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tflog", update_freq='batch')


# remote monitoring callback
monitor_callback = keras.callbacks.RemoteMonitor(root="http://localhost:8080",
                                                 path='/monitortraining')

history = model.fit(x=generator['train'], 
					steps_per_epoch=steps_per_epoch['train'],
					validation_data=generator['val'],
					validation_steps=steps_per_epoch['val'],
                    validation_freq=1, 
                    epochs=num_epochs, 
                    verbose=1,
                    callbacks=[saving_callback,
                               tfboard_callback,
                               monitor_callback])
history = history.history
accuracy = round(100 * max(history['accuracy']), 2)

# For log
_id = folder_name
_name = model_name
_training_date = today
_training_starttime = time_now
_training_stoptime = datetime.now().strftime("%H-%M-%S")
# get size of model
_size = getsize_h5model(checkpoint_path)
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

log_write(os.path.join(MODEL_PATH, LOG_FILE), log_content)

# after training delete the busy flag and raise the need_confirmed flage
os.remove(flag)
with open(os.path.join(MODEL_PATH, NEED_CONFIRM), 'wb') as f:
    pickle.dump({'need_confirm': 'yes'}, f)
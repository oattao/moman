import os
import sys
from datetime import date, datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.layers import Dropout, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation
from sklearn.model_selection import train_test_split

from utils.data import create_dataframe, ImageGenerator
from configs.image import IMAGE_SIZE2, IMAGE_SIZE3, DATA_PATH
from configs.server import BASE_MODELS, MODEL_PATH

import pdb

def load_model(models):
    print('Loading models...')
    model_list = dict()
    for name in models:
        model_list[name] = tf.keras.models.load_model(os.path.join(MODEL_PATH, name))
    print('All models loaded. Ready to serve.')
    return model_list

def create_model(model_name, n_outs):
    if model_name == 'Simple':
            model = Sequential([
            Conv2D(8, 3, padding='same', input_shape=IMAGE_SIZE3),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),
            
            Conv2D(16, 3, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),
            
            Conv2D(32, 3, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(64, 3, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),
            
            GlobalAveragePooling2D(),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(n_outs, activation='softmax')])

    if model_name == 'Mobilenet':
        model = Sequential([
            tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE3,
                                              include_top=False,
                                              weights='imagenet'),
            GlobalAveragePooling2D(),
            Dense(n_outs, activation='softmax')])

    if model_name == 'Xception':
        model = Sequential([
            tf.keras.applications.Xception(input_shape=IMAGE_SIZE3,
                                           include_top=False,
                                           weights='imagenet'),
            GlobalAveragePooling2D(),
            Dense(n_outs, activation='softmax')])    

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    return model

def create_checkpoint_path(model_name):
    today = date.today()
    time_now = datetime.now().strftime("%H-%M-%S")
    folder_name = '{}_{}_{}'.format(model_name, today, time_now)
    # checkpoint_path = os.path.join(MODEL_PATH, folder_name)
    return folder_name

def train_model(model_name, image_folder, num_epochs, learning_rate):
    # get number of classes (= number of subfolder)
    training_path = os.path.join(DATA_PATH, image_folder)
    classes = [name for name in os.listdir(training_path)\
         if os.path.isdir(os.path.join(training_path, name))] 
    num_classes = len(classes)

    # define model
    model = create_model(model_name, num_classes)

    # create image dataframe
    image_dataframe = create_dataframe(training_path, classes)

    # split train - test set
    data_split = ['train', 'test']
    train_frame, test_frame = train_test_split(image_dataframe, test_size=0.15, random_state=911)

    # image generator
    generator = {'train': ImageGenerator(df=train_frame, label_col='Classes', classes=classes),
                 'test': ImageGenerator(df=test_frame, label_col='Classes', classes=classes)}
    

    # checkpoint callbacks
    folder_name = create_checkpoint_path(model_name)
    checkpoint_path = os.path.join(MODEL_PATH, folder_name)
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

    return accuracy, history, folder_name

    





        
import os
import sys
from datetime import date, datetime
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.layers import Dropout, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation
from sklearn.model_selection import train_test_split

from utils.data import create_dataframe, ImageGenerator
from configs.image import IMAGE_SIZE2, IMAGE_SIZE3, DATA_PATH
from configs.server import BASE_MODELS, MODEL_PATH, MODEL_LOG, LOG_FILE

import pdb


def load_model(models):
    print('Loading models...')
    model_list = dict()
    for name in models:
        model_list[name] = tf.keras.models.load_model(
            os.path.join(MODEL_PATH, '{}.h5'.format(name)))
    print('All models loaded. Ready to serve.')
    return model_list


def create_model(model_name, n_outs):
    if model_name == 'Teachable_machine':
        # read the teachable machine
        base_model = tf.keras.models.load_model(os.path.join('.', 'utils', 'base_model.h5'))
        model = Sequential([base_model,
                            Dense(100, activation='relu'),
                            Dense(n_outs, activation='softmax')])

    if model_name == 'Tiny':
        model = Sequential([Flatten(), Dense(n_outs, activation='softmax')])
    if model_name == 'Small':
        model = Sequential([
            Conv2D(4, 3, padding='same', input_shape=IMAGE_SIZE3),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(n_outs, activation='softmax')])
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

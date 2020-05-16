import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.layers import Dropout, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation
sys.path.append(os.pardir)
from configs.image import RGB_IMAGE_SIZE3
from configs.server import SUPPORTED_MODELS

def load_model(model_path, h5_file_name='HKMold_AI.h5'):
    print('Loading models...')
    model_list = dict()
    for name in SUPPORTED_MODELS:
        if name != 'teachable_machine':
            model_list[name] = tf.keras.models.load_model(os.path.join(model_path, name))
        else:
            model_list[name] = tf.keras.models.load_model(os.path.join(model_path, name, h5_file_name))
    print('All models loaded. Ready to serve.')
    return model_list

def create_model(model_name):
    if model_name == 'simple':
            model = Sequential([
            Conv2D(8, 3, padding='same', input_shape=RGB_IMAGE_SIZE3),
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
            Dense(4, activation='softmax')])

    if model_name == 'mobilenet':
        model = Sequential([
            tf.keras.applications.MobileNetV2(input_shape=RGB_IMAGE_SIZE3,
                                              include_top=False,
                                              weights='imagenet'),
            GlobalAveragePooling2D(),
            Dense(4, activation='softmax')])

    if model_name == 'xception':
        model = Sequential([
            tf.keras.applications.Xception(input_shape=RGB_IMAGE_SIZE3,
                                           include_top=False,
                                           weights='imagenet'),
            GlobalAveragePooling2D(),
            Dense(4, activation='softmax')])    

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    return model
        
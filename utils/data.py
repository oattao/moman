import os
import sys
sys.path.append(os.pardir)

import pathlib
import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from PIL import Image, ImageOps

import matplotlib.pyplot as plt
from configs.image import IMAGE_FILE_EXTENSIONS

# from utils.server import TRAINING_FOLDER_ERROR

def verify_training_folder(data_path, image_folder):
    training_path = os.path.join(data_path, image_folder)

    # get sub folders:
    subfolders = [name for name in os.listdir(training_path) /
                    if os.path.isdir(training_path, name)] 
    if len(subfolders < 2):
        return 1
    for subf in subfolders:
        subf_path = pathlib.Path(os.path.join(training_path, subf))
        list_file = []
        for ext in IMAGE_FILE_EXTENSIONS:
            list_file.extend(list(subf_path.glob('*.{}'.format(os.path.sep, ext))))
            if len(list_file) == 0:
                return 2
    return 0

def show_batch(image_batch, label_batch):  
    plt.figure(figsize=(10,10))
    for n in range(8):
        ax = plt.subplot(2,4,n+1)
        plt.imshow(image_batch[n].squeeze())
        plt.title(np.array(QUALITY)[label_batch[n]==1][0].title().upper())
        plt.axis('off')

def create_image_dataframe(data_dir, file_extension='jpg'):
    cols = IMAGE_DATAFRAME_COLS
    data_dir = pathlib.Path(data_dir)
    list_file = list(data_dir.glob('*{}*.{}'.format(os.path.sep, file_extension)))

    image_dataframe = pd.DataFrame(columns=cols)
    for image_file in list_file:
        image_file = str(image_file)
        image_folder = image_file.split(os.path.sep)[-2]
        item = dict(zip(cols, [image_file, image_folder, LABELS[image_folder]]))
        item = pd.DataFrame(item, columns=cols, index=[0])
        image_dataframe = pd.concat((image_dataframe, item), axis=0, ignore_index=True)
    return image_dataframe

def show_prediction(model, image_path, image_show=True):
    # Get true value from image name
    # if image_show
    if image_show:
        img = plt.imread(image_path)
        plt.imshow(img)
    image = load_image(image_path=image_path, to_4d=True)
         
    prob = np.around(model.predict(image)[0] * 100, 1)
    prediction_proba = dict(zip(QUALITY, prob))
    pred_index = np.argmax(prob)
    pred = QUALITY[pred_index]
    print('Prediction quality: {}'.format(pred))
    print('Detail confident in percent (%):')
    for key in prediction_proba.keys():
        cofi = prediction_proba[key]
        print('Quality: {}, Confident: {:.1f}'.format(key, cofi))


class ImageGenerator(Sequence):
    """Generate data for training Tensorflow based models."""

    def __init__(self, df, label_col, image_shape=IMAGE_SIZE, batch_size=8, shuffle=True):
        """ Init function.
        Args:
            df (pandas DataFrame): contains time series data (equally scaled data).
            batch_size (int): number of items are generated in each batch.
            label_col
        """
        self.df = df
        self.label_col = label_col
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.df))
        self.shuffle = shuffle
        self.label = np.array(QUALITY)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):

        indexes = self.indexes[
            idx * self.batch_size:(idx + 1) * self.batch_size]
        x, y = [], []
        for i in indexes:

            file_path = self.df.iloc[i][IMAGE_DATAFRAME_COLS[0]]
            label = self.df.iloc[i][self.label_col] == self.label

            image = load_image(file_path, to_4d=False)

            x.append(image)
            y.append(label)
        x = np.array(x)
        y = np.array(y)
        return x, y, [None]

def load_image(image_path, to_4d, normalized=True, image_size=IMAGE_SIZE):
    image = Image.open(image_path).convert('RGB')
    image = ImageOps.fit(image, IMAGE_SIZE, Image.ANTIALIAS)
    image = np.asarray(image)
    if normalized:
        image = (image.astype(np.float32) / 127.5) - 1
    if to_4d:
        return np.array([image])
    else:
        return image

def get_sample(file_path, augumentation=False):
    # get label
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    label = parts[-2] == QUALITY

    # get image
    image_string = tf.io.read_file(file_path)
    image = tf.image.decode_image(image_string, channels=3, expand_animations=False)
    if augumentation:
        image = tf.image.random_contrast(image, lower=0.1, upper=0.6)
        image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    image = (image / 127.5) - 1

    return image, label

def config_batch(ds, cache=True, shuffle_buffer_size=100, batch_size=8):

    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

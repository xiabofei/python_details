# encoding=utf8

##################################################
# control randomness at the very beginning
##################################################
from numpy.random import seed
seed(2017)
from tensorflow import set_random_seed
set_random_seed(2017)
import random as rn
rn.seed(2017)
import os
os.environ['PYTHONHASHSEED'] = '0'


##################################################
# other imports
##################################################
import argparse
import numpy as np
import pandas as pd
import keras
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation
from keras.layers import LSTM
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback
from keras.applications import VGG16


from data_generator import AudioGenerator
from data_split import TRAIN_SPLIT_FILE_TEMP, VALID_SPLIT_FILE_TEMP, SPLIT_SEP
from fe_and_augmentation import LEGAL_LABELS
import pickle
import gc

from ipdb import set_trace as st

train_dir = '../data/input/processed_train/'
test_dir = '../data/input/processed_test/'


##################################################
# global parameters
##################################################
FLAGS = None
n_classes = len(LEGAL_LABELS)
RUNS_IN_FOLD = 3
batch_size = 64
epochs = 11


##################################################
# train and valid data generator
##################################################


##################################################
# load test data
##################################################
d_test = pickle.load(open(test_dir + 'test.pkl', 'rb'))
fname_test, X_test = d_test['fname'], d_test['data']
X_test = X_test.reshape(tuple(list(X_test.shape) + [1])).astype('float32')
del d_test
gc.collect()


##################################################
# define models
##################################################
def get_model():
    model = VGG16(include_top=True, weights=None, input_shape=(99, 161, 1), classes=n_classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model



##################################################
# callbacks
##################################################
def scheduler(epoch):
    return 0.001 if epoch<4 else 0.0001

class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = keras.backend.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))

lr_scheduler = LearningRateScheduler(scheduler)
lr_tracker = SGDLearningRateTracker()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str, default='0', help='which fold')
    FLAGS, _ = parser.parse_known_args()
    print('conduct train and test in fold {0}'.format(FLAGS.fold))
    train_generator = AudioGenerator(
        root_dir= '../data/input/train/audio/',
        k=FLAGS.fold,
        file_temp=TRAIN_SPLIT_FILE_TEMP,
        ori_batch_size=batch_size,
        train_or_valid='train',
        augmentation_prob=50,
    )
    train_generator.steps_per_epoch = train_generator.steps_per_epoch * 2
    valid_generator = AudioGenerator(
        root_dir= '../data/input/train/audio/',
        k=FLAGS.fold,
        file_temp=VALID_SPLIT_FILE_TEMP,
        ori_batch_size=batch_size,
        train_or_valid='valid',
    )
    preds = np.zeros((len(fname_test), n_classes))
    for run in range(RUNS_IN_FOLD):
        print('fold {0} runs {1}'.format(FLAGS.fold, run))
        model = get_model()
        model.fit_generator(
            generator=train_generator.generator(),
            steps_per_epoch=train_generator.steps_per_epoch,
            epochs=epochs,
            validation_data=valid_generator.generator(),
            validation_steps=valid_generator.steps_per_epoch,
            # callbacks=[lr_scheduler],
            shuffle=True,
        )
        gc.collect()
        preds += model.predict(X_test, batch_size=256) / RUNS_IN_FOLD
        del model
        gc.collect()

    labels_index =np.argmax(preds, axis=1)
    del X_test
    gc.collect()

    submit = pd.DataFrame()
    submit['fname'] = fname_test
    submit['label'] = [ LEGAL_LABELS[index] for index in labels_index]
    submit.to_csv('../data/output/submit_by_{0}fold.csv'.format(FLAGS.fold), index=False)
    print('train and test in fold {0} done'.format(FLAGS.fold))


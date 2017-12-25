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
from keras.layers import LSTM, Reshape, Permute, GRU
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import resnet50



from data_generator import AudioGenerator
from data_split import TRAIN_SPLIT_FILE_TEMP, VALID_SPLIT_FILE_TEMP, SPLIT_SEP
from fe_and_augmentation import LEGAL_LABELS
from fe_and_augmentation import SPEC, LMEL, MFCC, LFBANK
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
epochs = 50


##################################################
# train and valid data generator
##################################################

FE_TYPE = SPEC

##################################################
# load test data
##################################################
d_test = pickle.load(open(test_dir + 'test_{0}.pkl'.format(FE_TYPE), 'rb'))
fname_test, X_test = d_test['fname'], d_test['data']
X_test = X_test.reshape(tuple(list(X_test.shape) + [1])).astype('float32')
del d_test
gc.collect()


##################################################
# define models and feature extracting type
##################################################

def get_model():

    # input layer
    if FE_TYPE==SPEC:
        input_layer = Input(shape=(99, 161, 1), name='INPUT')
    if FE_TYPE==LFBANK:
        input_layer = Input(shape=(99, 40, 1), name='INPUT')
    layer = BatchNormalization()(input_layer)

    # conv1
    layer = Convolution2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = BatchNormalization()(layer)

    # conv2
    if FE_TYPE==SPEC:
        layer = Convolution2D(filters=32, kernel_size=(3,3), strides=(1,2), padding="same")(layer)
    if FE_TYPE==LFBANK:
        layer = Convolution2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = BatchNormalization()(layer)

    # conv3
    layer = Convolution2D(filters=64, kernel_size=(3,3), strides=(1,2), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = BatchNormalization()(layer)

    # conv4
    if FE_TYPE==SPEC:
        layer = Convolution2D(filters=128, kernel_size=(3,3), strides=(1,2), padding="same")(layer)
    if FE_TYPE==LFBANK:
        layer = Convolution2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(1,2))(layer)
    layer = BatchNormalization()(layer)


    layer = Reshape((12,128))(layer)

    # rnn
    layer = Bidirectional(LSTM(units=48, return_sequences=False))(layer)
    layer = Dropout(0.5)(layer)

    # fc1
    # layer = Flatten()(layer)
    # layer = Dense(units=512)(layer)
    # layer = Activation('relu')(layer)
    # layer = BatchNormalization()(layer)
    # layer = Dropout(0.5)(layer)

    # fc2
    # layer = Dense(units=256)(layer)
    # layer = Activation('relu')(layer)

    # output layer
    preds = Dense(units=n_classes, activation='softmax')(layer)

    # run through model
    model = Model(inputs=input_layer, outputs=preds)

    # model.compile(loss='categorical_hinge', optimizer='adam', metrics=['acc'])

    opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_hinge', optimizer=opt, metrics=['acc'])

    return model



##################################################
# callbacks
##################################################
def scheduler(epoch):
    lr = 0.01
    if epoch<=10:
        lr = 0.01
    elif epoch>10 and epoch<20:
        lr = 0.003
    elif epoch>20 and epoch<30:
        lr = 0.001
    elif epoch>30 and epoch<40:
        lr = 0.0003
    else:
        lr = 0.0001
    return lr

class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = keras.backend.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))

lr_scheduler = LearningRateScheduler(scheduler)
lr_tracker = SGDLearningRateTracker()
lr_plateau = ReduceLROnPlateau(monitor='val_acc', mode='max', patience=3, factor=0.3, verbose=1)


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
    # train_generator.steps_per_epoch = train_generator.steps_per_epoch * 2
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
        # use model check point callbacks
        bst_model_path = './tmp/nn_fold{0}_run{1}.h5'.format(FLAGS.fold, run)
        model_checkpoint = ModelCheckpoint(
            bst_model_path,
            monitor='val_acc',
            mode='max',
            save_best_only=True,
            save_weights_only=True
        )
        model = get_model()
        # st(context=21)
        hist = model.fit_generator(
            generator=train_generator.generator(FE_TYPE),
            steps_per_epoch=train_generator.steps_per_epoch,
            epochs=epochs,
            validation_data=valid_generator.generator(FE_TYPE),
            validation_steps=valid_generator.steps_per_epoch,
            callbacks=[lr_plateau, model_checkpoint],
            shuffle=True,
        )
        gc.collect()
        bst_acc = max(hist.history['val_acc'])
        print('best model val_acc : {0}'.format(bst_acc))
        model.load_weights(bst_model_path)
        os.remove(bst_model_path)
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


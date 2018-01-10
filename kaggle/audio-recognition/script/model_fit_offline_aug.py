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
from keras.layers import LSTM, Reshape, Permute, GRU, AveragePooling2D
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import SGD, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.applications import resnet50

from data_generator_aug_offline import AudioGenerator
from data_split import TRAIN_SPLIT_FILE_TEMP, VALID_SPLIT_FILE_TEMP, SPLIT_SEP
from fe_and_augmentation import LEGAL_LABELS
from fe_and_augmentation import SPEC, LMEL, MFCC, LFBANK
from fe_and_augmentation import conduct_fe
import pickle
import gc

from ipdb import set_trace as st

train_dir = '../data/input/processed_train/'
test_dir = '../data/input_backup/test_augmentation/'

##################################################
# global parameters
##################################################
FLAGS = None
n_classes = len(LEGAL_LABELS)
RUNS_IN_FOLD = 5
batch_size = 128
epochs = 60
FE_TYPE = SPEC



##################################################
# define models and feature extracting type
##################################################

def get_model():
    # input layer
    if FE_TYPE == SPEC:
        input_layer = Input(shape=(99, 161, 1), name='INPUT')
    if FE_TYPE == LFBANK:
        input_layer = Input(shape=(99, 40, 1), name='INPUT')
    layer = BatchNormalization()(input_layer)

    # conv1
    layer = Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = BatchNormalization()(layer)

    # conv2
    if FE_TYPE == SPEC:
        layer = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 2), padding="same")(layer)
    if FE_TYPE == LFBANK:
        layer = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = BatchNormalization()(layer)

    # conv3
    layer = Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 2), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = BatchNormalization()(layer)

    # conv4
    if FE_TYPE == SPEC:
        layer = Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 2), padding="same")(layer)
    if FE_TYPE == LFBANK:
        layer = Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(1, 2))(layer)
    layer = BatchNormalization()(layer)

    layer = Reshape((12, 128))(layer)

    # bi direction lstm
    # layer = Bidirectional(GRU(units=48, return_sequences=True))(layer)
    layer = Bidirectional(LSTM(units=48, return_sequences=True))(layer)
    layer = Bidirectional(LSTM(units=48, return_sequences=False))(layer)

    # bi direction gru
    # layer = Bidirectional(GRU(units=48, return_sequences=True))(layer)
    # layer = Bidirectional(GRU(units=48, return_sequences=False))(layer)

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


    opt = RMSprop(lr=1e-3, clipnorm=0.3)
    model.compile(loss='categorical_hinge', optimizer=opt, metrics=['acc'])
    # model.compile(loss='categorical_hinge', optimizer='adam', metrics=['acc'])
    #
    # opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_hinge', optimizer=opt, metrics=['acc'])

    return model


##################################################
# callbacks
##################################################

# learning rate scheduler
def scheduler(epoch):
    return 0.001 if epoch<10 else 1e-6
lr_scheduler = LearningRateScheduler(scheduler)

# learning rate plateau
lr_plateau = ReduceLROnPlateau(
    monitor='val_acc', mode='max',
    patience=3, cooldown=3,
    factor=np.sqrt(0.1), verbose=1, min_lr=1e-5, epsilon=8e-4)

# early stopping
early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=10)

# csv logger
# csv_logger  = CSVLogger('../data/output/log/train_log_{0}.csv'.format(FLAGS.fold))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str, default='0', help='which fold')
    FLAGS, _ = parser.parse_known_args()

    ##################################################
    # load test data
    ##################################################
    print('Prepare test data')
    d_test = pickle.load(open(test_dir + 'test_original.pkl', 'rb'))
    fname_test, X_test = d_test['fname'], d_test['data']
    X_test = X_test.reshape(tuple(list(X_test.shape) + [1])).astype('float32')
    del d_test
    gc.collect()

    ##################################################
    # create data generator
    ##################################################
    print('Prepare train data in fold {0}'.format(FLAGS.fold))
    train_generator = AudioGenerator(
        root_dir='../data/input_backup/train_augmentation/',
        k=FLAGS.fold,
        batch_size=batch_size,
        train_or_valid='train',
    )
    print('Prepare valid data in fold {0}'.format(FLAGS.fold))
    valid_generator = AudioGenerator(
        root_dir='../data/input_backup/train_augmentation/',
        k=FLAGS.fold,
        batch_size=batch_size,
        train_or_valid='valid',
    )
    # prepare valid data
    X_valid, y_valid = valid_generator.ori_data['data'], np.array(valid_generator.ori_data['label'])
    X_valid = X_valid.reshape(tuple(list(X_valid.shape) + [1])).astype('float32')
    del valid_generator
    gc.collect()

    preds_test = np.zeros((len(fname_test), n_classes))
    preds_valid = np.zeros((X_valid.shape[0], n_classes))

    ##################################################
    # train in folds
    ##################################################
    for run in range(RUNS_IN_FOLD):
        print('\nFold {0} runs {1}'.format(FLAGS.fold, run))
        # use model check point callbacks
        bst_model_path = '../data/output/model/nn_fold{0}_run{1}_{2}.h5'.format(FLAGS.fold, run, FE_TYPE)
        model_checkpoint = ModelCheckpoint(
            bst_model_path,
            monitor='val_acc',
            mode='max',
            save_best_only=True,
            save_weights_only=True
        )
        model = get_model()
        hist = model.fit_generator(
            generator=train_generator.generator(),
            steps_per_epoch=train_generator.steps_per_epoch,
            epochs=epochs,
            validation_data=(X_valid, y_valid),
            callbacks=[lr_plateau, model_checkpoint, early_stopping],
            shuffle=True,
        )
        train_generator.check_ori_and_aug_hits()
        gc.collect()
        bst_acc = max(hist.history['val_acc'])
        print('\nBest model val_acc : %.5f\n' % bst_acc)
        model.load_weights(bst_model_path)
        # os.remove(bst_model_path)
        # st(context=21)
        preds_test += model.predict(X_test, batch_size=batch_size) / RUNS_IN_FOLD
        preds_valid += model.predict(X_valid, batch_size=batch_size) / RUNS_IN_FOLD

        del model
        gc.collect()


    del X_test
    gc.collect()

    ##################################################
    # produce submit and valid result
    ##################################################
    labels_index_test = np.argmax(preds_test, axis=1)
    labels_index_valid = np.argmax(preds_valid, axis=1)

    # produce in fold predict file for CV
    in_fold = pd.DataFrame()
    in_fold['truth'] = [LEGAL_LABELS[index] for index in np.argmax(y_valid, axis=1)]
    in_fold['preds'] = [LEGAL_LABELS[index] for index in labels_index_valid]
    acc_counts = 0
    for truth, preds in zip(in_fold['truth'], in_fold['preds']):
        acc_counts += 1 if truth == preds else 0
    acc_in_fold = acc_counts / len(in_fold['preds'])
    print('\nFold %s valid accuracy : %.5f ' % (FLAGS.fold, acc_in_fold))
    in_fold.to_csv('../data/output/valid/valid_by_{0}fold_acc{1}.csv'.format(FLAGS.fold, acc_in_fold), index=False)

    # produce in fold submit file for ensemble
    submit = pd.DataFrame()
    submit['fname'] = fname_test
    submit['label'] = [LEGAL_LABELS[index] for index in labels_index_test]
    submit.to_csv('../data/output/submit/submit_by_{0}fold.csv'.format(FLAGS.fold), index=False)
    print('\nFold {0} done'.format(FLAGS.fold))

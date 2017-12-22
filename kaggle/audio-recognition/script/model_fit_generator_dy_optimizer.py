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
from keras.layers import TimeDistributed
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping


from data_generator import AudioGenerator
from data_split import TRAIN_SPLIT_FILE_TEMP, VALID_SPLIT_FILE_TEMP, SPLIT_SEP
from fe_and_augmentation import LEGAL_LABELS
from fe_and_augmentation import SPEC, MFCC, LFBANK, LMEL
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

epochs_stage1 = 25
opt_stage1 = Adam()

epochs_stage2 = 10
opt_stage2 = SGD(lr=0.001, momentum=0.9, nesterov=True)


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
FE_TYPE = LFBANK
def get_model():

    # input layer
    if FE_TYPE==SPEC:
        input_layer = Input(shape=(99, 161, 1), name='INPUT')
    if FE_TYPE==LFBANK:
        input_layer = Input(shape=(99, 40, 1), name='INPUT')
    layer = BatchNormalization()(input_layer)

    # conv1
    layer = Convolution2D(filters=8, kernel_size=(3,3), strides=(1,1), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = BatchNormalization()(layer)

    # conv2
    if FE_TYPE==SPEC:
        layer = Convolution2D(filters=16, kernel_size=(3,3), strides=(1,2), padding="same")(layer)
    if FE_TYPE==LFBANK:
        layer = Convolution2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = BatchNormalization()(layer)

    # conv3
    layer = Convolution2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = BatchNormalization()(layer)

    # conv4
    layer = Convolution2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = BatchNormalization()(layer)

    layer = Flatten()(layer)

    # fc1
    layer = Dense(units=512)(layer)
    layer = Activation('relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)

    # fc2
    layer = Dense(units=256)(layer)
    layer = Activation('relu')(layer)

    # output layer
    preds = Dense(units=n_classes, activation='softmax')(layer)

    # run through model
    model = Model(inputs=input_layer, outputs=preds)

    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # opt = SGD(lr=0.001, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

    return model



##################################################
# callbacks
##################################################
def scheduler(epoch):
    return 0.001 if epoch<6 else 0.0003

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

    ## train and valid data generator
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

    ## train runs in fold
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

        # train stage 1 use 'adam' as optimizer for fast convergence
        print('fold {0} runs {1} : train stage 1 begin'.format(FLAGS.fold, run))
        model_stage1 = get_model()
        model_stage1.compile(loss='categorical_hinge', optimizer=opt_stage1, metrics=['acc'])
        hist = model_stage1.fit_generator(
            generator=train_generator.generator(),
            steps_per_epoch=train_generator.steps_per_epoch,
            epochs=epochs_stage1,
            validation_data=valid_generator.generator(),
            validation_steps=valid_generator.steps_per_epoch,
            callbacks=[model_checkpoint,],
            shuffle=True,
        )
        del model_stage1
        gc.collect()
        bst_acc = max(hist.history['val_acc'])
        print('stage 1 best model val_acc : {0}'.format(bst_acc))

        # train stage 2 use 'sgd' as optimizer for fine-tuning
        print('fold {0} runs {1} : train stage 2 begin'.format(FLAGS.fold, run))
        model_stage2 = get_model()
        model_stage2.compile(loss='categorical_hinge', optimizer=opt_stage2, metrics=['acc'])
        print('fold {0} runs {1} : load best model weights from stage 1'.format(FLAGS.fold, run))
        model_stage2.load_weights(bst_model_path)
        os.remove(bst_model_path)
        hist = model_stage2.fit_generator(
            generator=train_generator.generator(),
            steps_per_epoch=train_generator.steps_per_epoch,
            epochs=epochs_stage2,
            validation_data=valid_generator.generator(),
            validation_steps=valid_generator.steps_per_epoch,
            callbacks=[lr_scheduler, model_checkpoint,],
            shuffle=True,
        )
        bst_acc = max(hist.history['val_acc'])
        print('stage 2 best model val_acc : {0}'.format(bst_acc))
        model_stage2.load_weights(bst_model_path)


        # predict in run of fold
        preds += model_stage2.predict(X_test, batch_size=256) / RUNS_IN_FOLD

        # clean scene in this run
        del model_stage2
        os.remove(bst_model_path)
        gc.collect()

    labels_index =np.argmax(preds, axis=1)
    del X_test
    gc.collect()

    # create submit file
    submit = pd.DataFrame()
    submit['fname'] = fname_test
    submit['label'] = [ LEGAL_LABELS[index] for index in labels_index]
    submit.to_csv('../data/output/submit_by_{0}fold.csv'.format(FLAGS.fold), index=False)
    print('train and test in fold {0} done'.format(FLAGS.fold))

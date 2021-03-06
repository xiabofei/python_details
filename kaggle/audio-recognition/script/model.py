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
import numpy as np
import pandas as pd
import keras
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback


from fe_and_augmentation import LEGAL_LABELS
import pickle
import gc

from ipdb import set_trace as st

train_dir = '../data/input/processed_train/'
test_dir = '../data/input/processed_test/'


##################################################
# global parameters
##################################################
n_classes = len(LEGAL_LABELS)
k = 0
batch_size = 64
epochs = 10


##################################################
# load processed train and test data
##################################################
d_train = pickle.load(open(train_dir + '{0}_fold_'.format(k) + 'train.pkl', 'rb'))
d_valid = pickle.load(open(train_dir + '{0}_fold_'.format(k) + 'valid.pkl', 'rb'))
d_test = pickle.load(open(test_dir + 'test.pkl', 'rb'))

y_train, X_train = d_train['label'], d_train['data']
y_valid, X_valid = d_valid['label'], d_valid['data']
fname_test, X_test = d_test['fname'], d_test['data']


X_train = X_train.reshape(tuple(list(X_train.shape) + [1])).astype('float32')
X_valid = X_valid.reshape(tuple(list(X_valid.shape) + [1])).astype('float32')
X_test = X_test.reshape(tuple(list(X_test.shape) + [1])).astype('float32')

y_train = to_categorical(y_train, n_classes)
y_valid = to_categorical(y_valid, n_classes)

del d_train
del d_valid
del d_test
gc.collect()

print('X_train shape:{0}'.format(X_train.shape))
print('X_valid shape:{0}'.format(X_valid.shape))


##################################################
# define models
##################################################
def get_model():

    # input layer
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2], 1), name='INPUT')
    layer = BatchNormalization()(input_layer)

    # first conv-conv-pooling layer group
    layer = Convolution2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = Convolution2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = Dropout(rate=0.25)(layer)

    # second conv-conv-pooling layer group
    layer = Convolution2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = Convolution2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = Dropout(rate=0.25)(layer)
    layer = Flatten()(layer)

    # fc
    layer = BatchNormalization()(layer)
    layer = Dense(units=256, activation='relu')(layer)
    layer = Dropout(0.5)(layer)

    # output layer
    preds = Dense(units=n_classes, activation='softmax')(layer)

    # whole model
    model = Model(inputs=input_layer, outputs=preds)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # opt = SGD(lr=0.001, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

    return model

model = get_model()


##################################################
# train process
##################################################

# callbacks
def scheduler(epoch):
    return 0.001 if epoch<3 else 0.0001

lr_scheduler = LearningRateScheduler(scheduler)

class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = keras.backend.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))

lr_tracker = SGDLearningRateTracker()

model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_valid, y_valid),
    callbacks=[lr_tracker,lr_scheduler],
    shuffle=True
)

del X_train
del X_valid
gc.collect()
##################################################
# test process
##################################################
preds = model.predict(X_test, batch_size=256)
labels_index =np.argmax(preds, axis=1)

# # input layer
# model_input = Model(inputs=model.inputs, outputs=model.get_layer(name='INPUT').output)
# input_ori = model_input.predict(X_test)
# # internal BN layer 0
# model_bn_0 = Model(inputs=model.inputs, outputs=model.get_layer(name='BN0').output)
# bn_output_0 = model_bn_0.predict(X_test)
# # internal BN layer 1
# model_bn_1 = Model(inputs=model.inputs, outputs=model.get_layer(name='BN0').output)
# bn_output_1 = model_bn_1.predict(X_test)

del X_test
gc.collect()


##################################################
#  create submit file
##################################################
submit = pd.DataFrame()
submit['fname'] = fname_test
submit['label'] = [ LEGAL_LABELS[index] for index in labels_index]
submit.to_csv('../data/output/submit_by_{0}fold.csv'.format(k), index=False)


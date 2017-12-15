# encoding=utf8

from numpy.random import seed
seed(2017)
from tensorflow import set_random_seed
set_random_seed(2017)
import random as rn
rn.seed(2017)
import os
os.environ['PYTHONHASHSEED'] = '0'

##################################################
# control randomness at the very beginning
##################################################



from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation
from keras.models import Model
from keras.utils import to_categorical

from fe_and_augmentation import LEGAL_LABELS
import pickle




from ipdb import set_trace as st

root_dir = '../data/input/processed_train/'

n_classes = len(LEGAL_LABELS)
k = 0
batch_size = 64
epochs = 5

d_train = pickle.load(open('../data/input/processed_train/{0}_fold_'.format(k) + 'train.pkl', 'rb'))
d_valid = pickle.load(open('../data/input/processed_train/{0}_fold_'.format(k) + 'valid.pkl', 'rb'))

y_train, X_train = d_train['label'], d_train['data']
y_valid, X_valid = d_valid['label'], d_valid['data']


X_train = X_train.reshape(tuple(list(X_train.shape) + [1])).astype('float32')
X_valid = X_valid.reshape(tuple(list(X_valid.shape) + [1])).astype('float32')
y_train = to_categorical(y_train, n_classes)
y_valid = to_categorical(y_valid, n_classes)
# st(context=21)

print('X_train shape:{0}'.format(X_train.shape))
print('X_valid shape:{0}'.format(X_valid.shape))
##################################################
# define models
##################################################
def get_model():

    # input layer
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2], 1))
    layer = BatchNormalization()(input_layer)

    # first conv layer
    layer = Convolution2D(filters=16, kernel_size=(2,3), strides=(1,1))(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = Dropout(rate=0.2)(layer)

    # second conv layer
    layer = Convolution2D(filters=16, kernel_size=(2,3), strides=(1,1))(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = Dropout(rate=0.2)(layer)
    layer = Flatten()(layer)

    # fc
    layer = BatchNormalization()(layer)
    layer = Dense(units=128, activation='relu')(layer)

    # output layer
    preds = Dense(units=n_classes, activation='softmax')(layer)

    model = Model(inputs=input_layer, outputs=preds)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

model = get_model()

model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_valid, y_valid),
    shuffle=True
)






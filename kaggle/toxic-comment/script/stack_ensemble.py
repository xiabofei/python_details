# encoding=utf8
from data_split import K
from keras.layers import Input
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras import Model
from data_split import label_candidates
from comm_preprocessing import ID_COL
import pandas as pd
import numpy as np
from numpy import hstack
from keras.optimizers import Nadam, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dropout
import gc

from ipdb import set_trace as st

n_classes = len(label_candidates)

rnn = '../data/output/stack_ensemble/rnn/'
cnn = '../data/output/stack_ensemble/cnn/'
lr = '../data/output/stack_ensemble/lr/'

stack_ensemble_materials = [rnn,cnn,lr]

EPOCHS = 20
RUNS = 2
BATCH_SIZE = 128


EPS = 1e-8

def get_ensemble_inputShape():
    ret = (len(stack_ensemble_materials) * n_classes, )
    return ret

def get_model():
    '''stack ensemble by NN model
    '''
    input_layer = Input(shape=get_ensemble_inputShape())
    layer = Dense(units=128, activation='relu')(input_layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.3)(layer)
    output_layer = Dense(6, activation='sigmoid')(layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model

def re_range(data):
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + EPS)
    data = (data - 0.5) * 2
    return data

def get_data():

    # get test submit data
    test_submit = []
    id_test = []
    for solo_path in stack_ensemble_materials:
        df = pd.read_csv(solo_path + 'avg_submit.csv')
        preds = df[label_candidates].values
        id_test = df[ID_COL].values.tolist()
        test_submit.append(preds)
    test_submit = hstack(test_submit)
    test_submit = re_range(test_submit)

    # get id and true label
    df_train = pd.read_csv('../data/input/train.csv')
    ids = df_train[ID_COL].values.tolist()
    y_true = df_train[label_candidates].values

    # get all model oof valid preds score
    preds_score = []
    fold_ids = dict() # get ids in each fold
    for solo_path in stack_ensemble_materials:
        id_preds = {}
        for k in range(K):
            df = pd.read_csv(solo_path + '{0}fold_valid.csv'.format(k))
            id_list = df[ID_COL].values.tolist()
            fold_ids[k] = id_list
            preds_list = df[label_candidates].values.tolist()
            for id, preds in zip(id_list, preds_list):
                id_preds[id] = preds
        y_score = np.array([id_preds[id] for id in ids])
        preds_score.append(y_score)
    preds_score = hstack(np.array(preds_score))
    preds_score = re_range(preds_score)

    idx_trn_val = []
    for k in range(K):
        ids_val = set(fold_ids[k])
        ids_trn = []
        for key in fold_ids.keys():
            if key==k:
                continue
            ids_trn += fold_ids[key]
        ids_trn = set(ids_trn)
        idx_trn = [idx for idx, id in enumerate(ids) if id in ids_trn]
        idx_val = [idx for idx, id in enumerate(ids) if id in ids_val]
        idx_trn_val.append((idx_trn, idx_val))
    return test_submit, id_test, preds_score, y_true, idx_trn_val, fold_ids,


if __name__ == '__main__':
    test_submit, id_test, preds_score, y_true, idx_trn_val, fold_ids = get_data()

    preds_test = np.zeros((test_submit.shape[0], n_classes))
    for fold, (idx_trn, idx_val) in enumerate(idx_trn_val):
        preds_valid = np.zeros((len(idx_val), n_classes))

        for run in range(RUNS):
            print('Fold {0} runs {1}'.format(fold, run))

            model = get_model()

            es = EarlyStopping(monitor='val_acc', mode='max', patience=6)
            bst_model_path = '../data/output/model/ensemble_bst_model.h5'
            mc = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
            rp = ReduceLROnPlateau(
                monitor='val_acc', mode='max',
                patience=4,
                factor=np.sqrt(0.1),
                verbose=1
            )

            hist = model.fit(
                x=preds_score[idx_trn,:], y=y_true[idx_trn,:],
                validation_data=(preds_score[idx_val,:], y_true[idx_val,:]),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                callbacks=[es, mc, rp]
            )
            model.load_weights(bst_model_path)
            bst_val_score = max(hist.history['val_acc'])
            print('\nBest val score : {0}'.format(bst_val_score))

            # predict
            preds_test += model.predict(test_submit, batch_size=1024, verbose=1) / RUNS / K
            preds_valid += model.predict(preds_score[idx_val,:], batch_size=1024, verbose=1) / RUNS

            del model
            gc.collect()

        preds_valid = preds_valid.T
        df_preds_val = pd.DataFrame()
        df_preds_val[ID_COL] = fold_ids[fold]
        for idx, label in enumerate(label_candidates):
            df_preds_val[label] = preds_valid[idx]
        df_preds_val.to_csv('../data/output/stack_ensemble/{0}fold_valid.csv'.format(fold), index=False)

    # record ensemble result
    preds_test = preds_test.T
    df_preds_test = pd.DataFrame()
    df_preds_test[ID_COL] = id_test
    for idx, label in enumerate(label_candidates):
        df_preds_test[label] = preds_test[idx]
    df_preds_test.to_csv('../data/output/stack_ensemble/ensemble_submit.csv', index=False)


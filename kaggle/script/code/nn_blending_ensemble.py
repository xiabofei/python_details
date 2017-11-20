import os
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

###############################################
## define models
###############################################

def get_model():
    input_layer = Input(shape=(train_x.shape[1],))
    
    layer = Dense(np.random.randint(384, 512))(input_layer)
    layer = BatchNormalization()(layer)
    layer = PReLU()(layer)
    layer = Dropout(0.5)(layer)

    layer = Dense(np.random.randint(128, 192))(layer)
    layer = BatchNormalization()(layer)
    layer = PReLU()(layer)
    layer = Dropout(0.5)(layer)

    preds = Dense(1, activation='sigmoid')(layer)

    model = Model(inputs=input_layer, outputs=preds)
    
    model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['acc'])
    return model

###############################################
## validate the ensemble model
###############################################

#np.random.seed(10)

NBAGS = 6
NFOLDS = 8
if(re_weight):
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

from sklearn.model_selection import KFold
kf = KFold(n_splits=NFOLDS, shuffle=True)

cv_pred = np.zeros((len(train_x), 1))
test_pred = np.zeros((len(test_x), 1))
for i in range(NBAGS):
    print('------------------------------')
    print('  Bag %d begins'%i)
    print('------------------------------')
    count = 0
    for trn_index, val_index in kf.split(range(train_x.shape[0])):
        print('  CV fold %d begins'%count)

        trn_index = np.random.permutation(trn_index)
        trn_x, val_x = train_x[trn_index], train_x[val_index]
        trn_y, val_y = train_y[trn_index], train_y[val_index]
        
        early_stopping =EarlyStopping(monitor='val_loss', patience=10)
        bst_model_path = './tmp/blending_nn.h5'
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        
        model = get_model()
        hist = model.fit(trn_x, trn_y,
                         validation_split = 0.1, verbose=0, \
            epochs=100, batch_size=4096, shuffle=True, class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])
            
        model.load_weights(bst_model_path)
        os.remove(bst_model_path)
        bst_score_pret = min(hist.history['val_loss'])
        print('     Fold %d finished, val-loss: %.5f'%(count, bst_score_pret))
        
        cv_pred[val_index] += model.predict(val_x, batch_size=4096, verbose=0) / NBAGS
        test_pred += model.predict(test_x, batch_size=32768, verbose=0) / NBAGS / NFOLDS
        
        count += 1


cv_loss = weighted_log_loss(train_y, cv_pred, re_weight)
print('------------------------------')
print('CV prediction finished, CV loss: %.5f'%cv_loss)
print('------------------------------')

###############################################
## make prediction
###############################################

submission = pd.DataFrame({'test_id': test_id, 'is_duplicate': test_pred.ravel()})
submission.to_csv('./%.5f_submission_nn_%s.csv'%(cv_loss, re_weight), index=False)


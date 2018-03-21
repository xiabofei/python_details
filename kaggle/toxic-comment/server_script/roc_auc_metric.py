from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

VAL_AUC = 'val_auc'

class RocAucMetricCallback(Callback):
    def __init__(self, predict_batch_size=1024, include_on_batch=False):
        super(RocAucMetricCallback, self).__init__()
        self.predict_batch_size=predict_batch_size
        self.include_on_batch=include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if(self.include_on_batch):
            logs[VAL_AUC]=float('-inf')
            if(self.validation_data):
                logs[VAL_AUC]=roc_auc_score(
                    self.validation_data[1],
                    self.model.predict(self.validation_data[0], batch_size=self.predict_batch_size)
                )

    def on_train_begin(self, logs={}):
        if not (VAL_AUC in self.params['metrics']):
            self.params['metrics'].append(VAL_AUC)

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        logs[VAL_AUC]=float('-inf')
        if(self.validation_data):
            logs[VAL_AUC]=roc_auc_score(
                self.validation_data[1],
                self.model.predict(self.validation_data[0], batch_size=self.predict_batch_size))

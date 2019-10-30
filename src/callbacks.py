import os
from os.path import dirname, abspath
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, Callback, ReduceLROnPlateau


def build_callbacks(model_path, model_name):
    abs_path = dirname(dirname(abspath(__file__)))
    history = CSVLogger(abs_path + model_path + model_name + '.log', separator=',', append=False)

    check_point = ModelCheckpoint(os.path.join(model_path, model_name + ".h5"),
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='min',)

    early_stopping = EarlyStopping(patience=10,
                                   verbose=1,
                                   monitor='val_loss',
                                   mode='min')

    reduce_lr = ReduceLROnPlateau(factor=0.5,
                                  patience=5,
                                  min_lr=0.000001,
                                  verbose=1,
                                  monitor='val_loss',
                                  mode='max')

    callbacks = [check_point, early_stopping, reduce_lr]
    return callbacks


# inheritance for training process plot 
class PlotLearning(Callback):

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        # self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        # image_val_dir = '../dataset/images_val'
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('mean_iou'))
        self.val_acc.append(logs.get('val_mean_iou'))
        self.i += 1
        print('i=', self.i, 'loss=', logs.get('loss'), 'val_loss=', logs.get('val_loss'), 'mean_iou=', logs.get('mean_iou'), 'val_mean_iou=', logs.get('val_mean_iou'))


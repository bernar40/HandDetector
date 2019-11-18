import os
from os.path import dirname, abspath
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau


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

    callbacks = [history, early_stopping, reduce_lr]
    return callbacks
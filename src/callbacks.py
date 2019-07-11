import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os.path import dirname, abspath
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import keras


def build_callbacks(model_name):
    abs_path = dirname(dirname(abspath(__file__)))
    checkpointer = ModelCheckpoint(filepath='unet.h5', verbose=0, save_best_only=True, save_weights_only=True)
    history = CSVLogger(abs_path + '/models/' + model_name + '.log', separator=',', append=False)
    callbacks = [checkpointer, PlotLearning(), history]
    return callbacks


# inheritance for training process plot 
class PlotLearning(keras.callbacks.Callback):

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

        """
        # choose a random test image and precess
        path = np.random.choice(image_val_dir)
        raw = Image.open(f'{path}')
        raw = np.array(raw.resize((256, 256)))/255.
        
        # predict the mask
        pred = model.predict(np.expand_dims(raw, 0))
        
        # mask post-processing
        msk  = pred.squeeze()
        msk = np.stack((msk,)*3, axis=-1)
        msk[msk >= 0.5] = 1 
        msk[msk < 0.5] = 0 
        
        # show the mask and the segmented image
        combined = np.concatenate([raw, msk, raw* msk], axis = 1)
        plt.axis('off')
        plt.imshow(combined)
        plt.show()
        """
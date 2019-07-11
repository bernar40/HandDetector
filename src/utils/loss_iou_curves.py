import matplotlib.pyplot as plt
import pandas as pd


def show_iou_loss(history=None, log=None):
    if history or log:
        if log:
            history = pd.read_csv(log, sep=',', engine='python')
        acc = history['mean_iou']
        val_acc = history['val_mean_iou']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, '-b', label='Training acc')
        plt.plot()
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, '-b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
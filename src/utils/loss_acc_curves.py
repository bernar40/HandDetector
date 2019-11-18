import matplotlib.pyplot as plt
import pandas as pd


def show_acc_loss(history=None, log=None, acc='accuracy'):
    if history or log:
        if log:
            history = pd.read_csv(log, sep=',', engine='python')
        acc = history[acc]
        val_acc = history['val_' + acc]
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, '-b', label='Training IOU')
        plt.plot(epochs, val_acc, 'r', label='Validation IOU')
        plt.title('Training and validation IOU')
        plt.ylabel('IOU')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        plt.plot(epochs, loss, '-b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
import matplotlib.pyplot as plt
import pandas as pd


def show_iou_loss(history=None, log=None):
    if history or log:
        if log:
            history = pd.read_csv(log, sep=',', engine='python')
        iou = history['iou_score']
        val_iou = history['val_iou_score']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(iou) + 1)

        plt.plot(epochs, iou, '-b', label='Training IOU')
        plt.plot(epochs, val_iou, 'r', label='Validation IOU')
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
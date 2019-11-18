from matplotlib import pyplot as plt
import numpy as np


def show(generators):
    for gen in generators:

        x, y = next(gen)
        print(x[0].shape, y[0].shape)
        fig = plt.figure(figsize=(16, 16))
        for i, (img, mask) in enumerate(zip(x, y)):
            fig.add_subplot(1, 8, i + 1)
            plt.imshow(img)
        plt.show()
        fig = plt.figure(figsize=(16, 16))
        for i, (img, mask) in enumerate(zip(x, y)):
            fig.add_subplot(1, 8, i + 1)
            plt.imshow(np.squeeze(mask, axis=-1))
        plt.show()
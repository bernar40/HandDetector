from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import keras
import cv2

def get_segmentation_generator_flow_dir(image_dir, mask_dir, target_size, batch_size, datagen_args=None, seed=2019,
                               shuffle=True):
    # datagen_args = dict(rescale=1./255,featurewise_center=False)  --> exemplo de datagen_args
    # usado no argumento da funcao

    if datagen_args is None:
        image_datagen = ImageDataGenerator(rescale=1. / 255)
        mask_datagen = ImageDataGenerator(rescale=1. / 255)
    else:  # For data augmentation
        image_datagen = ImageDataGenerator(**datagen_args)
        mask_datagen = ImageDataGenerator(**datagen_args)

    image_generator = image_datagen.flow_from_directory(
        image_dir,
        class_mode=None,
        classes=['.'],
        seed=seed,
        batch_size=batch_size,
        shuffle=shuffle,
        target_size=target_size,
        color_mode="rgb")

    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        class_mode=None,
        classes=['.'],
        seed=seed,
        batch_size=batch_size,
        shuffle=shuffle,
        target_size=target_size,
        color_mode="grayscale")

    # Just zip the two generators to get a generator that provides augmented images and masks at the same time
    train_generator = zip(image_generator, mask_generator)

    return train_generator


def get_segmentation_generator_flow(images, masks, target_size, batch_size, datagen_args=None, seed=810,
                               shuffle=True):

    # datagen_args = dict(rescale=1./255,featurewise_center=False)  --> exemplo de datagen_args
    # usado no argumento da funcao

    if datagen_args is None:
        image_datagen = ImageDataGenerator(rescale=1. / 255)
        mask_datagen = ImageDataGenerator(rescale=1. / 255)
    else:  # For data augmentation
        image_datagen = ImageDataGenerator(**datagen_args)
        mask_datagen = ImageDataGenerator(**datagen_args)

    image_generator = image_datagen.flow(
        images,
        seed=seed,
        batch_size=batch_size,
        shuffle=shuffle)

    mask_generator = mask_datagen.flow(
        masks,
        seed=seed,
        batch_size=batch_size,
        shuffle=shuffle)

    # Just zip the two generators to get a generator that provides augmented images and masks at the same time
    train_generator = zip(image_generator, mask_generator)

    return train_generator


class DataGenerator(keras.utils.Sequence):
    def __init__(self, imgs, masks, preprocess_input, aug,
                 batch_size=32,
                 dim=(256, 256), shuffle=True
                 ):
        # 'Initialization'
        self.X = imgs
        self.M = masks
        self.batch_size = batch_size
        self.n_classes = 1
        self.shuffle = shuffle
        self.preprocess_input = preprocess_input
        self.aug = aug
        self.on_epoch_end()
        self.dim = dim

    def __len__(self):
        return int(np.floor((len(self.X) / self.batch_size) / 1))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__data_generation(indexes)

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        batch_size = len(indexes)

        # Initialization
        xx = np.empty((batch_size, self.dim[0], self.dim[1], 3), dtype='float32')
        yy = np.empty((batch_size, self.dim[0], self.dim[1], 1), dtype='float32')

        # Resize Data
        for i, ID in enumerate(indexes):
            img = self.X[ID]
            if img.shape[0] != self.dim[0]:
                img = cv2.resize(img, self.dim, cv2.INTER_CUBIC)
            mask = self.M[ID]
            if mask.shape[0] != self.dim[0]:
                mask = cv2.resize(mask, self.dim, cv2.INTER_AREA)

            # Augment Images
            augmented = self.aug(image=img, mask=mask)
            aug_img = augmented['image']
            aug_mask = augmented['mask']
            aug_mask = np.expand_dims(aug_mask, axis=-1)
            aug_mask = aug_mask / 255

            assert (np.max(aug_mask) <= 1.0 and np.min(aug_mask) >= 0)
            aug_mask[aug_mask > 0.5] = 1
            aug_mask[aug_mask < 0.5] = 0

            xx[i, ] = aug_img.astype('float32')
            yy[i, ] = aug_mask.astype('float32')

        xx = self.preprocess_input(xx)

        return xx, yy
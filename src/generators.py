from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image


def get_segmentation_generator(image_dir, mask_dir, target_size, batch_size, datagen_args=None, seed=2019, shuffle=True):
    
    # datagen_args = dict(rescale=1./255,featurewise_center=False)  --> exemplo de datagen_args
    # usado no argumento da funcao
    
    if datagen_args is None:
        image_datagen = ImageDataGenerator(rescale=1./255)
        mask_datagen = ImageDataGenerator(rescale=1./255)
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
                target_size=(256, 256),
                color_mode="grayscale")

    # Just zip the two generators to get a generator that provides augmented images and masks at the same time
    train_generator = zip(image_generator, mask_generator)
    
    return train_generator


def image_generator(files, path_to_mask, path_to_file, batch_size = 32, sz = (256, 256)):
  
    while True:
        # extract a random batch
        batch = np.random.choice(files, size = batch_size)

        # variables for collecting batches of inputs and outputs
        batch_x = []
        batch_y = []

        for f in batch:

            # get the masks. Note that masks are png files
            mask = Image.open(f'{path_to_mask}{f[:-4]}.png')
            mask = np.array(mask.resize(sz))

            # preprocess the mask
            mask[mask >= 2] = 0
            mask[mask != 0 ] = 1

            batch_y.append(mask)

            # preprocess the raw images
            raw = Image.open(f'{path_to_file}{f}')
            raw = raw.resize(sz)
            raw = np.array(raw)

            # check the number of channels because some of the images are RGBA or GRAY
            if len(raw.shape) == 2:
                raw = np.stack((raw,)*3, axis=-1)
            else:
                raw = raw[:,:,0:3]

            batch_x.append(raw)

        # preprocess a batch of images and masks
        batch_x = np.array(batch_x)/255.
        batch_y = np.array(batch_y)
        batch_y = np.expand_dims(batch_y,3)

        yield (batch_x, batch_y)
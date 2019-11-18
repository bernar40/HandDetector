from keras.preprocessing.image import ImageDataGenerator


def get_segmentation_generator_flow_dir(image_dir, mask_dir, target_size, batch_size, datagen_args=None, seed=810,
                                        shuffle=True):

    # datagen_args = dict(rescale=1./255,featurewise_center=False)

    if datagen_args is None:
        image_datagen = ImageDataGenerator(rescale=1. / 255)
        mask_datagen = ImageDataGenerator()
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


def get_segmentation_generator_flow(images, masks, batch_size, datagen_args=None, seed=810,
                                    shuffle=True):

    # datagen_args = dict(rescale=1./255,featurewise_center=False)

    if datagen_args is None:
        image_datagen = ImageDataGenerator(rescale=1. / 255)
        mask_datagen = ImageDataGenerator()
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

from src.models import *
from src.generators import *
from src.callbacks import *
from src.utils.file_paths import *
from src.utils.loss_acc_curves import *


batch_size = 32
target_size = (256, 256)

image_dir = 'dataset/images/'
mask_dir = 'dataset/masks/'
img_val_dir = 'dataset/images_val/'
mask_val_dir = 'dataset/masks_val/'
model_name = 'hands_1'
image_files = absoluteFilePaths(image_dir)
mask_files = absoluteFilePaths(mask_dir)

move_files_back_to_all(image_dir, mask_dir, img_val_dir, mask_val_dir)

split = int(0.95 * len(image_files))

# split into training and testing
img_train_files = image_files[0:split]
img_test_files = image_files[split:]
mask_train_files = mask_files[0:split]
mask_test_files = mask_files[split:]

move_files_to_val(img_test_files, mask_test_files, img_val_dir, mask_val_dir)

train_generator = get_segmentation_generator(image_dir, mask_dir, target_size, batch_size)
test_generator = get_segmentation_generator(img_val_dir, mask_val_dir, target_size, batch_size)

model = unet()

train_steps = len(img_train_files) // batch_size
test_steps = len(mask_train_files) // batch_size


model.fit_generator(train_generator,
                    epochs=30,
                    steps_per_epoch=train_steps,
                    validation_data=test_generator,
                    validation_steps=test_steps,
                    callbacks=build_callbacks(model_name),
                    verbose=0)

show_acc_loss(log=f'models/{model_name}')
model.save(f'models/{model_name}')
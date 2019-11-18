from src.models import *
from src.callbacks import *
from src.utils.file_paths import *
from src.utils.loss_acc_curves import *
from src.generators import *
import cv2
import h5py
import numpy as np
import segmentation_models as sm


batch_size = 8
target_size = (256, 256)

model_name = 'hands'
BACKBONE = 'resnet18'
model_name = model_name + BACKBONE

model_dir = 'models/'

aug = dict(rescale=1./255,featurewise_center=False, rotation_range=40,
           zoom_range=0.3, width_shift_range=0.2, height_shift_range=0.2,
           shear_range=0.15,horizontal_flip=True, fill_mode="nearest")

with h5py.File("dataset/Training/bgr_comp_hand_segmentation_data.h5",
               "r") as hdf:
    data = hdf.get("images")
    images = np.array(data)

    data2 = hdf.get("masks")
    annotations = np.array(data2)

# model = unet2(input_size=(128, 128, 3))

model = sm.Unet(BACKBONE, classes=1, input_shape=(256, 256, 3))

# model = sm.FPN(backbone_name=BACKBONE, encoder_weights='imagenet',
#                activation='sigmoid', classes=1, pyramid_dropout=0.5)

model.compile(optimizer=Adam(lr=0.0001), loss=sm.losses.jaccard_loss, metrics=[mean_iou])

train_steps = len(images) // batch_size
test_steps = len(annotations) // batch_size


train_generator = get_segmentation_generator_flow(images[:750], annotations[:750],
                                                  target_size, batch_size,
                                                  datagen_args=aug, shuffle=True)
test_generator = get_segmentation_generator_flow(images[750:], annotations[750:],
                                                 target_size, batch_size,
                                                 shuffle=False)


history = model.fit_generator(train_generator,
                              epochs=30,
                              steps_per_epoch=train_steps,
                              validation_data=test_generator,
                              validation_steps=test_steps,
                              callbacks=build_callbacks(model_dir, model_name),
                              verbose=1)

plt.plot(history.history['mean_iou'])
plt.plot(history.history['val_mean_iou'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

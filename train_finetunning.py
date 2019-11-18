from src.models import *
from src.callbacks import *
from src.utils.show_images import *
from src.utils.loss_iou_curves import *
from src.generators import *
import cv2
import h5py
import datetime
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from matplotlib import pyplot as plt
from keras.preprocessing import image
from PIL import Image as im

batch_size = 8
target_size = (256, 256)

model_name = 'hands_finetunning_'
BACKBONE = 'inceptionresnetv2'
model_name = model_name + BACKBONE

model_dir = '/content/drive/My Drive/TCC/models/'

aug = dict(rescale=1./255,featurewise_center=False, rotation_range=40,
           zoom_range=0.3, width_shift_range=0.2, height_shift_range=0.2,
           shear_range=0.15,horizontal_flip=True, fill_mode="nearest")

total_epochs = 40
epoch_1phase = total_epochs // 2  # training only the final layers
epoch_2phase = total_epochs - epoch_1phase  # training all layers -- increse the performance.

with h5py.File("dataset/Training/comp_hand_segmentation_data.h5",
               "r") as hdf:
    data = hdf.get("images")
    images = np.array(data)

    data2 = hdf.get("masks")
    annotations = np.array(data2)

model = sm.Unet(backbone_name=BACKBONE, encoder_weights='imagenet', encoder_freeze=True, input_shape=(256, 256, 3))
model.compile('Adam', loss=sm.losses.jaccard_loss, metrics=[sm.metrics.iou_score])

train_steps = len(images) // batch_size
test_steps = len(annotations) // batch_size

train_size = round(len(images)*0.7)
valid_size = train_size + round(len(images)*0.2)

train_images = images[:train_size]
train_annotations = annotations[:train_size]

valid_images = images[train_size:valid_size]
valid_annotations = annotations[train_size:valid_size]

test_images = images[valid_size:]
test_annotations = annotations[valid_size:]

train_generator = get_segmentation_generator_flow(train_images, train_annotations,
                                                  target_size, batch_size,
                                                  datagen_args=aug, shuffle=True)

valid_generator = get_segmentation_generator_flow(valid_images, valid_annotations,
                                                  target_size, batch_size,
                                                  shuffle=False)

test_generator = get_segmentation_generator_flow(test_images, test_annotations,
                                                 target_size, batch_size,
                                                 shuffle=False)

show([train_generator, valid_generator, test_generator])

history = model.fit_generator(train_generator,
                              epochs=epoch_1phase,
                              steps_per_epoch=train_steps,
                              validation_data=valid_generator,
                              validation_steps=test_steps,
                              verbose=1)

history_stats = history.history
epoch_1phase_end = len(history_stats['val_loss'])
sm.utils.set_trainable(model)

history = model.fit_generator(train_generator,
                              initial_epoch=epoch_1phase_end,
                              epochs=epoch_1phase_end + epoch_2phase,
                              steps_per_epoch=train_steps,
                              validation_data=test_generator,
                              validation_steps=test_steps,
                              callbacks=build_callbacks(model_dir, model_name),
                              verbose=1)

show_iou_loss(history=history)

time_now = str(datetime.datetime.now())
model.save('/content/drive/My Drive/TCC/models/' + model_name + '_' + time_now + '.h5')

converter = tf.lite.TFLiteConverter.from_keras_model_file(model_dir + model_name, custom_objects={"jaccard_loss": sm.losses.jaccard_loss, "mean_iou": mean_iou})
tflite_model = converter.convert()
open("/content/converted_model.tflite", "wb").write(tflite_model)


batch_holder_x = np.zeros((8, 256, 256, 3))
batch_holder_y = np.zeros((8, 256, 256, 3))

batch_holder_x, batch_holder_y = next(test_generator)

print(batch_holder_x.shape)

fig = plt.figure(figsize=(8, 8))

for i,img in enumerate(batch_holder_x):
  fig.add_subplot(4, 4, i+1)
  plt.imshow(img)

plt.show()

results = model.predict(batch_holder_x)
threshold = 0.50
for res in results:
  res[res >= threshold] = 1
  res[res < threshold] = 0

fig = plt.figure(figsize=(8, 8))

print("results")
for i,img in enumerate(results):
  fig.add_subplot(4, 4, i+1)
  plt.imshow(np.squeeze(results[i], axis=-1))

plt.show()

fig = plt.figure(figsize=(8, 8))


print("Groundtruth")
for i,img in enumerate(batch_holder_y):
  fig.add_subplot(4, 4, i+1)
  plt.imshow(np.squeeze(batch_holder_y[i], axis=-1))

plt.show()

img_dir='/content/test_imgs/'
batch_holder = np.zeros((8, 256, 256, 3))
for i,img in enumerate(os.listdir(img_dir)):
  if 'ipynb' in img:
    continue
  img = im.open(os.path.join(img_dir,img))
  img = img.resize((256, 256))
  img  = np.array(img)
  batch_holder[i, :] = img

from keras.preprocessing.image import ImageDataGenerator
image_datagen = ImageDataGenerator(rescale=1. / 255)
image_generator = image_datagen.flow_from_directory(
      img_dir,
      class_mode=None,
      classes=['.'],
      seed=2018,
      batch_size=8,
      shuffle=True,
      target_size=(256, 256),
      color_mode="rgb")

results=model.predict(batch_holder)

threshold = 0.50
for res in results:
  res[res >= threshold] = 1
  res[res < threshold] = 0

batch_holder = next(image_generator)
fig = plt.figure(figsize=(8, 8))

for i,img in enumerate(batch_holder):
  fig.add_subplot(4, 4, i+1)
  plt.imshow(img)

plt.show()

fig = plt.figure(figsize=(8, 8))

print("results")
for i,img in enumerate(results):
  fig.add_subplot(4, 4, i+1)
  plt.imshow(np.squeeze(results[i], axis=-1))

plt.show()
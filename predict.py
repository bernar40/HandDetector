from src.generators import *
import cv2
import os
import numpy as np
from keras.models import *
from segmentation_models.metrics import iou_score
from segmentation_models.losses import jaccard_loss
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image as im
import matplotlib.pyplot as plt


model_dir = 'models/'
model_name = 'hands_finetunning_inceptionresnetv2_2019-11-17 22_47_47.042294.h5'

model = load_model(model_dir + model_name, custom_objects={"jaccard_loss": jaccard_loss, "iou_score": iou_score})

with h5py.File("dataset/Training/comp_hand_segmentation_data.h5", "r") as hdf:
    data = hdf.get("images")
    images = np.array(data)

    data2 = hdf.get("masks")
    annotations = np.array(data2)

train_size = round(len(images)*0.7)
valid_size = train_size + round(len(images)*0.2)

test_images = images[valid_size:]
test_annotations = annotations[valid_size:]

test_generator = get_segmentation_generator_flow(test_images, test_annotations, 8, shuffle=False)

batch_holder_x = np.zeros((8, 256, 256, 3))
batch_holder_y = np.zeros((8, 256, 256, 3))

batch_holder_x, batch_holder_y = next(test_generator)

fig = plt.figure(figsize=(8, 8))

for i, img in enumerate(batch_holder_x):
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
for i, img in enumerate(results):
    fig.add_subplot(4, 4, i+1)
    plt.imshow(np.squeeze(results[i], axis=-1))

plt.show()

fig = plt.figure(figsize=(8, 8))


print("Groundtruth")
for i, img in enumerate(batch_holder_y):
    fig.add_subplot(4, 4, i+1)
    plt.imshow(np.squeeze(batch_holder_y[i], axis=-1))

plt.show()

img_dir = 'dataset/Test/Real Images/'
batch_holder = np.zeros((8, 256, 256, 3))
for i, img in enumerate(os.listdir(img_dir)):
    if 'ipynb' in img:
        continue
    img = im.open(os.path.join(img_dir, img))
    img = img.resize((256, 256))
    img = np.array(img)
    batch_holder[i, :] = img

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


batch_holder = next(image_generator)

fig = plt.figure(figsize=(8, 8))
for i, img in enumerate(batch_holder):
    fig.add_subplot(4, 4, i+1)
    plt.imshow(img)
plt.show()

results = model.predict(batch_holder)
threshold = 0.50
for res in results:
    res[res >= threshold] = 1
    res[res < threshold] = 0

fig = plt.figure(figsize=(8, 8))
print("results")
for i, img in enumerate(results):
    fig.add_subplot(4, 4, i+1)
    plt.imshow(np.squeeze(results[i], axis=-1))
plt.show()

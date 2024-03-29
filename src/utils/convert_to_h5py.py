import numpy as np
import h5py
import os
from PIL import Image as im
# from matplotlib import pyplot as plt


image_dir = '../../dataset/Training/Image/'
mask_dir = '../../dataset/Training/Annotation/'

image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

xx = np.empty((899, 256, 256, 3), dtype='float32')
yy = np.empty((899, 256, 256, 1), dtype='float32')

for i, mask_file in enumerate(mask_files):
    img_file = mask_file.split('.')[0] + '.jpg'
    img_file = img_file.replace('Annotation', 'Image')

    img = im.open(img_file)
    img = img.resize((256, 256))
    img = np.array(img)

    mask = im.open(mask_file, )
    mask = mask.resize((256, 256)).convert('L')
    mask = np.array(mask)
    mask = np.expand_dims(mask, axis=-1)

    xx[i,] = img.astype('float32')
    yy[i,] = mask.astype('float32')

# Create a new HDF5 file
with h5py.File("/Users/bernardoruga/Documents/PUC/TCC/HandDetector/dataset/Training/hand_segmentation_data.h5", "w") as hdf:
    # Create a dataset in the file
    hdf.create_dataset("images", data=xx)
    hdf.create_dataset("masks", data=yy)

# Create a new HDF5 file
with h5py.File("/Users/bernardoruga/Documents/PUC/TCC/HandDetector/dataset/Training/comp_hand_segmentation_data.h5", "w") as hdf:
    # Create a dataset in the file
    hdf.create_dataset("images", data=xx, compression="gzip", compression_opts=9)
    hdf.create_dataset("masks", data=yy, compression="gzip", compression_opts=9)

#
# with h5py.File("/Users/bernardoruga/Documents/PUC/TCC/HandDetector/dataset/Training/bgr_comp_hand_segmentation_data.h5",
#                "r") as hdf:
#     print(hdf.keys)
#     data = hdf.get("images")
#     dataset1 = np.array(data)
#
#     data2 = hdf.get("masks")
#     dataset2 = np.array(data2)
#     print(dataset1.shape)
#
# from matplotlib import pyplot as plt
#
# plt.imshow(dataset1[0].astype(int))
# plt.show()
#
# a = np.squeeze(dataset2[0], axis=-1)
# plt.imshow(a)
# plt.show()

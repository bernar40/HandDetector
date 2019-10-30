from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import shutil


def convert_mask_2_binary(mask_dir, new_masks):
	for mask in mask_dir:
		col = Image.open(mask) # read image
		gray = col.convert('L')  # conversion to gray scale
		bw = gray.point(lambda x: 0 if x<128 else 255, '1')  # binarization
		bw.save("test_bw.png") # save it

		col = Image.open("test_bw.png")
		bw = np.invert(np.asarray(col))
		mask = mask.split('/')[-1]
		plt.imsave(f'{new_masks}{mask}' , np.array(bw), cmap=cm.gray)
	os.remove("test_bw.png")



img_dir = "/Users/bernardoruga/PycharmProjects/HandDetector/original_images/"
mask_dir = "/Users/bernardoruga/PycharmProjects/HandDetector/final_masks/"

mask_dst = "/Users/bernardoruga/PycharmProjects/HandDetector/Masks_New/"
imgs_dst = "/Users/bernardoruga/PycharmProjects/HandDetector/Images_New/"


imgs = [f for f in listdir(img_dir) if isfile(join(img_dir, f)) and f.endswith('.jpg') or f.endswith('.JPG')]
masks = [mask_dir + f for f in listdir(mask_dir) if isfile(join(mask_dir, f)) and f.endswith('.bmp')]

imgs = sorted(imgs)
masks = sorted(masks)

for mask in masks:
	os.rename(mask, mask.split('.')[0] + '.png')

for mask, img in zip(masks,imgs):
	if i < 10:
		m_dst = mask_dst + '00' + str(i) + '.bmp'
		i_dst = imgs_dst + '00' + str(i) + '.jpg'
	elif 9 < i < 100:
		m_dst = mask_dst + '0' + str(i) + '.bmp'
		i_dst = imgs_dst + '0' + str(i) + '.jpg'
	else:
		m_dst = mask_dst + str(i) + '.bmp'
		i_dst = imgs_dst + str(i) + '.jpg'

	shutil.copy(mask_dir + mask, m_dst)
	shutil.copy(img_dir + img, i_dst)
	i += 1



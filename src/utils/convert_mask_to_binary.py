from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os


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

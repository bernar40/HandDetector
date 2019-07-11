import os
import shutil


def absoluteFilePaths(directory):
	all_files = []
	for dirpath, _, filenames in os.walk(directory):
		for f in filenames:
			all_files.append(os.path.abspath(os.path.join(dirpath, f)))
	return all_files


def move_files_back_to_all(img_destination, masks_destination, image_val_dir, mask_val_dir):
	image_vals = absoluteFilePaths(image_val_dir)
	mask_vals = absoluteFilePaths(mask_val_dir)
	for image_val, mask_val in zip(image_vals, mask_vals):
		shutil.move(image_val, img_destination)
		shutil.move(mask_val, masks_destination)


def move_files_to_val(img_test_files, mask_test_files, img_destination, masks_destination):
	for file, mask in zip(img_test_files, mask_test_files):
		shutil.move(file, img_destination)
		shutil.move(mask, masks_destination)

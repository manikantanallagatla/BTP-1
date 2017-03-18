#importing libraries
import cv2
import numpy as np
import scipy
from scipy import ndimage
from scipy.misc import toimage
import matplotlib.pyplot as plt

def removeImage(file_input):
	#read image
	img = cv2.imread(file_input, cv2.IMREAD_GRAYSCALE)
	# img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	#trying grayscale
	label_img, cc_num = ndimage.label(img)
	CC = ndimage.find_objects(label_img)
	cc_areas = ndimage.sum(img, label_img, range(cc_num+1))
	area_mask = (cc_areas > 300)
	label_img[area_mask[label_img]] = 255
	toimage(label_img).show()

removeImage('english_with_image.png')
import numpy as np
import cv2

import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from skimage import color
import glob


def filterer():
    for img_file in glob.glob("*.jpg"):
        # print(str(img_file)[:-4] +".jpg")
        img = plt.imread(img_file)
        # img_orig_show = color.rgb2gray(io.imread(img_file))

        filtered_img_name = str(img_file)[:-4] + "_bilateral_filter.jpg"
        bilateral_filt = cv2.bilateralFilter(img, 9, 75, 75)
        plt.imsave(filtered_img_name, bilateral_filt)
        multi_filt = color.rgb2gray(io.imread(filtered_img_name))

        io.imshow(multi_filt)
        io.imsave(filtered_img_name, multi_filt)


# filterer(cv2.bilateralFilter(img,9, 75, 75))
#filterer()
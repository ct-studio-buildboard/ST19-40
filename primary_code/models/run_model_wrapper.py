
from models import unet_with_gan
from models import helper_functions
from models.unet_with_gan import BASIC_D
from models.unet_with_gan import UBlock
from models.unet_with_gan import UNET_G
from models.unet_with_gan import netG_gen
from models.unet_with_gan import netG_train
from models.unet_with_gan import netD_train

from models.unet_with_gan import weights_init
from models.unet_with_gan import Dataset_Handler

from models.helper_functions import process_img
from models.helper_functions import process_img2
from models.helper_functions import read_img_simult

from models.helper_functions import show_image
import time

import scipy.misc
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import glob
from random import randint, shuffle
import torchvision
import torchvision.transforms as transforms
import torch
import torchvision
import torchvision.transforms as transforms

import torch
import torchvision
import torchvision.transforms as transforms

from torchvision import transforms, datasets
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as data_torch
from sklearn.metrics import balanced_accuracy_score
from skimage import io, transform

nc_in = 3
nc_out = 3
ngf = 64
ndf = 64

loadSize = 286
imageSize = 256
batchSize = 1
lrD = 2e-4
lrG = 2e-4

class Model_Handler(object):

    def __init__(self,path_gen):
        netG = UNET_G(imageSize, nc_in, nc_out, ngf)
        netG.load_state_dict(torch.load(path_gen, map_location='cpu'))

        self.model=netG

    def predict(self, img_path):
        temp_image=None
        alt_right = np.zeros((3, 256, 256))
        try:
            temp_image = process_img2(img_path)
            alt_right[0] = temp_image.numpy()
            alt_right[1] = temp_image.numpy()
            alt_right[2] = temp_image.numpy()
            alt_right = torch.Tensor(alt_right)
        except:
            pass

        try:
            temp_image = process_img(img_path)
            alt_right=temp_image
            alt_right = torch.Tensor(alt_right)
        except:
            pass




        temp_whut = netG_gen(alt_right.reshape(1, 3, 256, 256), self.model)

        return temp_whut

if __name__ == '__main__':
    path_gen = "../full_models/gen_D"



    model=Model_Handler(path_gen)
    temp_whut=model.predict("../sample_images/calibri_a.jpg")

    #show_image(temp_whut[0].T)


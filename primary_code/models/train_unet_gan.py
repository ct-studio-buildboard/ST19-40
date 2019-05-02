from startup_studio.models import unet_with_gan
from startup_studio.models import helper_functions
from startup_studio.models.unet_with_gan import BASIC_D
from startup_studio.models.unet_with_gan import UBlock
from startup_studio.models.unet_with_gan import UNET_G
from startup_studio.models.unet_with_gan import netG_gen
from startup_studio.models.unet_with_gan import netG_train
from startup_studio.models.unet_with_gan import netD_train

from startup_studio.models.unet_with_gan import weights_init
from startup_studio.models.unet_with_gan import Dataset_Handler

from startup_studio.models.helper_functions import process_img
from startup_studio.models.helper_functions import process_img2
from startup_studio.models.helper_functions import read_img_simult

from startup_studio.models.helper_functions import show_image
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
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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


def get_all_img_pairings(paths):
  elements=[]
  for i in paths[0:]:
    print(i)
    elements.append(read_img_simult(i[0],i[1]))
  return elements

def V(x):
    return Variable(x).cuda()

if __name__ == '__main__':





    #AGGREGATE ALL THE DATA
    english = "MyDrive/deep_learning_characters_for_analysis/characters/english"
    greek = "MyDrive/deep_learning_characters_for_analysis/characters/greek"
    english_v = "MyDrive/deep_learning_characters_for_analysis/val_characters/english"
    greek_v = "MyDrive/deep_learning_characters_for_analysis/val_characters/greek"

    print(os.path.isdir("MyDrive/deep_learning_characters_for_analysis/characters/english"))

    english_file_paths = []
    greek_file_paths = []
    english_file_paths_val = []
    greek_file_paths_val = []

    english_paths = helper_functions.getFiles(english)
    greek_paths = helper_functions.getFiles(greek)
    english_paths.sort()
    greek_paths.sort()

    english_and_greek_paths = []
    english_and_greek_paths_v = []

    for i in range(len(english_paths)):
        english_and_greek_paths.append((english_paths[i], greek_paths[i]))

    english_paths_val = helper_functions.getFiles(english_v)
    greek_paths_val = helper_functions.getFiles(greek_v)
    english_paths_val.sort()
    greek_paths_val.sort()



    for i in range(len(english_paths_val)):
        english_and_greek_paths_v.append((english_paths_val[i], greek_paths_val[i]))

    #FORMAT TRAINING DATA

    training_data = get_all_img_pairings(english_and_greek_paths)
    val_data = get_all_img_pairings(english_and_greek_paths_v)

    for_data_loader = Dataset_Handler(training_data)
    for_val_loader = Dataset_Handler(val_data)
    trainloader = torch.utils.data.DataLoader(for_data_loader, batch_size=20, shuffle=False, num_workers=0)
    valloader = torch.utils.data.DataLoader(for_val_loader, batch_size=20, shuffle=False, num_workers=0)

    trainloader_all = torch.utils.data.DataLoader(for_data_loader, batch_size=3000, shuffle=False, num_workers=0)


    ####Build MODEL

    nc_in = 3
    nc_out = 3
    ngf = 64
    ndf = 64


    loadSize = 286
    imageSize = 256
    batchSize = 1
    lrD = 2e-4
    lrG = 2e-4

    netD = BASIC_D(nc_in, nc_out, ndf)
    netD.apply(weights_init)

    netG = UNET_G(imageSize, nc_in, nc_out, ngf)
    netG.apply(weights_init)

    optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999))

    history_val = []  # so i can see how it develops over time
    history_train = []  # so i can see how it develops over time




    t0 = time.time()
    niter = 30
    gen_iterations = 0
    errL1 = epoch = errG = 0
    errL1_sum = errG_sum = errD_sum = 0

    display_iters = 1
    val_batch = iter(valloader)  # minibatch(english_and_greek_paths_v, 1, direction)
    train_batch = iter(trainloader)  # minibatch(english_and_greek_paths, 1, direction)
    epoch = 0

    first_time = True
    while epoch < niter:
        trainA = None
        trainB = None
        try:
            trainA, trainB = next(train_batch)
        except:
            epoch += 1
            trainloader = torch.utils.data.DataLoader(for_data_loader, batch_size=20, shuffle=True, num_workers=1)
            train_batch = iter(trainloader)
            trainA, trainB = next(train_batch)
            first_time = True

        print(epoch)
        print(len(trainA))
        print(len(trainB))
        vA, vB = V(trainA), V(trainB)
        print(vA.shape)
        print(vB.shape)
        errD, = netD_train(vA, vB,netD,netG,optimizerD)
        errD_sum += errD

        # epoch, trainA, trainB = next(train_batch)
        errG, errL1 = netG_train(vA, vB,netD,netG,optimizerG)
        errG_sum += errG
        errL1_sum += errL1
        gen_iterations += 1

        if epoch % display_iters == 0:
            if first_time == True:


                channel_first = 0


                trainloaderULT = torch.utils.data.DataLoader(for_data_loader, batch_size=180, shuffle=False) #so can vsiualize it all

                vA, vB = iter(trainloaderULT).next()
                fakeB = netG_gen(V(vA),netG)
                history_train.append(fakeB)  # store history



                first_time = False
                errL1_sum = errG_sum = errD_sum = 0
                val_batch = iter(valloader)
                valA, valB = next(val_batch)
                fakeB = netG_gen(V(valA),netG)
                print(type(vA))
                print(type(vB))
                print(type(fakeB))

                history_val.append(fakeB)
                show_image(valA[0].cpu().numpy().T)
                show_image(valB[0].cpu().numpy().T)
                show_image(fakeB[0].T)

                show_image(valA[1].cpu().numpy().T)
                show_image(valB[1].cpu().numpy().T)
                show_image(fakeB[1].T)



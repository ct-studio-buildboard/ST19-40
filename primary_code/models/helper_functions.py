

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
from skimage import io, transform

channel_first=True
channel_axis=1
nc_in = 3
nc_out = 3
ngf = 64
ndf = 64

loadSize = 286
imageSize = 256



def show_image(image):
    """Show image with landmarks"""
    #plt.savefig("a.png")
    plt.imshow(image)
    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)


def showX(X, rows=1):
    assert X.shape[0]%rows == 0
    int_X = ( (X+1)/2*255).clip(0,255).astype('uint8')
    if channel_first:
        int_X = np.moveaxis(int_X.reshape(-1,3,imageSize,imageSize), 1, 3)
    else:
        int_X = int_X.reshape(-1,imageSize,imageSize, 3)
    int_X = int_X.reshape(rows, -1, imageSize, imageSize,3).swapaxes(1,2).reshape(rows*imageSize,-1, 3)
    #display(Image.fromarray(int_X))


def getFiles(root):
    file_paths = []
    print(root)
    for path, subdirs, files in os.walk(root):

        for name in files:
            if (".DS_Store" in name):
                pass
            else:
                file_paths.append(os.path.join(path, name))
    return file_paths

def process_img(name_a):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    image = io.imread(name_a)
    im = Image.fromarray(np.uint8(image))
    return preprocessing(im)


def process_img2(name_a):
    normalize = transforms.Normalize(mean=[0.485],
                                     std=[0.229])
    preprocessing = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    image = io.imread(name_a)
    im = Image.fromarray(np.uint8(image))
    return preprocessing(im)


def read_img_simult(name_a, name_b):
    a = process_img(name_a)
    b = process_img(name_b)

    return (a, b)


def select_training_and_validation(all_path_pairs,data,font_name,letter_number):
    new_train = []
    new_val = []
    font = "Bahnschrift_Light"
    number = ""
    for paths, associated_data in zip(all_path_pairs, data):
        # print("OK")
        # print(paths)
        # print(len(associated_data))
        if font in paths[0]:
            if number in paths[0] or number == "":
                new_val.append(associated_data)
        else:
            if number in paths[0] or number == "":
                new_train.append(associated_data)
    return new_train,new_val

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

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as data_torch
from sklearn.metrics import balanced_accuracy_score
from skimage import io, transform

one = None
zero = None

def conv_block(in_feat, out_feat, ksize, stride, padding,
               activation=nn.LeakyReLU(0.2, inplace=True), use_batchnorm=True):
    layers = [nn.Conv2d(in_feat, out_feat, ksize, stride, padding, bias=not use_batchnorm)]
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_feat))
    if activation:
        layers.append(activation)
    return nn.Sequential(*layers)


class BASIC_D(nn.Module):
    def __init__(self, nc_in, nc_out, ndf, max_layers=3):
        super(BASIC_D, self).__init__()
        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial_{0}-{1}'.format(nc_in + nc_out, ndf),
                        conv_block(nc_in + nc_out, ndf, 4, 2, 1, use_batchnorm=False))
        out_feat = ndf
        for layer in range(1, max_layers):
            in_feat = out_feat
            out_feat = ndf * min(2 ** layer, 8)
            main.add_module('pyramid_{0}-{1}'.format(in_feat, out_feat),
                            conv_block(in_feat, out_feat, 4, 2, 1, ))
        in_feat = out_feat
        out_feat = ndf * min(2 ** max_layers, 8)
        main.add_module('last_{0}-{1}'.format(in_feat, out_feat),
                        conv_block(in_feat, out_feat, 4, 1, 1))

        in_feat, out_feat = out_feat, 1
        main.add_module('output_{0}-{1}'.format(in_feat, out_feat),
                        conv_block(in_feat, out_feat, 4, 1, 1, nn.Sigmoid(), False))
        self.main = main

    def forward(self, a, b):
        x = torch.cat((a, b), 1)
        output = self.main(x)
        return output

class UBlock(nn.Module):
    def __init__(self, s, nf_in, max_nf, use_batchnorm=True, nf_out=None, nf_next=None):
        super(UBlock, self).__init__()
        assert s>=2 and s%2==0
        nf_next = nf_next if nf_next else min(nf_in*2, max_nf)
        nf_out = nf_out if nf_out else nf_in
        self.conv = nn.Conv2d(nf_in, nf_next, 4, 2, 1, bias=not (use_batchnorm and s>2) )
        if s>2:
            next_block = [nn.BatchNorm2d(nf_next)] if use_batchnorm else []
            next_block += [nn.LeakyReLU(0.2, inplace=True), UBlock(s//2, nf_next, max_nf)]
            self.next_block = nn.Sequential(*next_block)
        else:
            self.next_block = None
        convt = [nn.ReLU(),
                 nn.ConvTranspose2d(nf_next*2 if self.next_block else nf_next, nf_out,
                                        kernel_size=4, stride=2,padding=1, bias=not use_batchnorm)]
        if use_batchnorm:
            convt += [nn.BatchNorm2d(nf_out)]
        if s <= 8:
            convt += [nn.Dropout(0.5, inplace=True)]
        self.convt = nn.Sequential(*convt)

    def forward(self, x):
        x = self.conv(x)
        if self.next_block:
            x2 = self.next_block(x)
            x = torch.cat((x,x2),1)
        return self.convt(x)


def UNET_G(isize, nc_in=3, nc_out=3, ngf=64):
    return nn.Sequential(
                  UBlock(isize, nc_in, 8*ngf, False, nf_out=nc_out, nf_next=ngf),
                  nn.Tanh() )



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Dataset_Handler(data_torch.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data=data #ASSUMES TUPLES WITH FIRSST BEING INPUT, SECOND BEING PREDICTED CLASS


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.data[index][0],self.data[index][1]

  loss = nn.BCELoss()
  lossL1 = nn.L1Loss()
  one = None
  zero = None

def netD_train(A, B, netD,netG, optimizerD):
    loss = nn.BCELoss()
    lossL1 = nn.L1Loss()
    global one, zero
    netD.zero_grad()
    output_D_real = netD(A, B)
    if one is None:
      one = Variable(torch.ones(*output_D_real.size())).cuda()
    errD_real = loss(output_D_real, one)
    errD_real.backward()

    output_G = netG(A)
    output_D_fake = netD(A, output_G)
    if zero is None:
      zero = Variable(torch.zeros(*output_D_fake.size())).cuda()
    errD_fake = loss(output_D_fake, zero)
    errD_fake.backward()
    optimizerD.step()

    return (errD_fake.data + errD_real.data) / 2,

def netG_train(A, B, netD,netG, optimizerG):
    loss = nn.BCELoss()
    lossL1 = nn.L1Loss()
    global one
    netG.zero_grad()
    output_G = netG(A)
    output_D_fake = netD(A, output_G)
    if one is None:
      one = Variable(torch.ones(*output_D_fake.size())).cuda()
    errG_fake = loss(output_D_fake, one)
    errG_L1 = lossL1(output_G, B)
    errG = errG_fake + 100 * errG_L1
    errG.backward()

    optimizerG.step()
    print(errG_fake.data)
    print(errG_L1.data)
    return errG_fake.data, errG_L1.data


def netG_gen(A, netG):
    return np.concatenate([netG(A[i:i+1]).data.cpu().numpy() for i in range(A.size()[0])], axis=0)
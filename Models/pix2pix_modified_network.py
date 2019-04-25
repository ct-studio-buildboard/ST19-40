# -*- coding: utf-8 -*-
"""pix2pix-torch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19lQ0nGwlqGd42O5cdzl8V4QatnSYytwq

#Create drive refence
"""

# Install a Drive FUSE wrapper.
# https://github.com/astrada/google-drive-ocamlfuse
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse



# Generate auth tokens for Colab
from google.colab import auth
auth.authenticate_user()


# Generate creds for the Drive FUSE library.
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}


# Create a directory and mount Google Drive using that directory.
!mkdir -p MyDrive
!google-drive-ocamlfuse MyDrive


!ls MyDrive/

# Create a file in Drive.
!echo "This newly created file will appear in your Drive file list." > MyDrive/created.txt

import os 
os.path.isfile("MyDrive/deep_learning_characters_for_analysis/characters/english/Arial_Black/english_Arial_Black_0.jpg")

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

import numpy as np 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as data_torch
from sklearn.metrics import balanced_accuracy_score
from skimage import io, transform

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
        main.add_module('initial_{0}-{1}'.format(nc_in+nc_out, ndf),
                        conv_block(nc_in+nc_out, ndf, 4, 2, 1, use_batchnorm=False))
        out_feat = ndf
        for layer in range(1, max_layers):
            in_feat = out_feat
            out_feat = ndf * min(2**layer, 8)
            main.add_module('pyramid_{0}-{1}'.format(in_feat, out_feat),
                                conv_block(in_feat, out_feat, 4, 2, 1, ))           
        in_feat = out_feat
        out_feat = ndf*min(2**max_layers, 8)
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

"""#Dataset handler"""

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

"""#Instatiate network"""

nc_in = 3
nc_out = 3
ngf = 64
ndf = 64
λ = 10

loadSize = 286
imageSize = 256
batchSize = 1
lrD = 2e-4
lrG = 2e-4

netD = BASIC_D(nc_in, nc_out, ndf)
netD.apply(weights_init)

print("ok")

netG = UNET_G(imageSize, nc_in, nc_out, ngf)
netG.apply(weights_init)

print("Ok")

inputA = torch.FloatTensor(batchSize, nc_in, imageSize, imageSize)
inputB = torch.FloatTensor(batchSize, nc_out, imageSize, imageSize)

# load data 
# 
#
from os import listdir
from os.path import isfile, join



"""#Create files for font"""

import sys,os
#MyDrive/deep_learning_characters_for_analysis/
english="MyDrive/deep_learning_characters_for_analysis/characters/english"
greek="MyDrive/deep_learning_characters_for_analysis/characters/greek"


print(os.path.isdir("MyDrive/deep_learning_characters_for_analysis/characters/english"))

english_file_paths=[]
greek_file_paths=[]

def getFiles(root):
    file_paths=[]
    print(root)
    for path, subdirs, files in os.walk(root):
       
        for name in files:
            if (".DS_Store" in name): pass 
            else:
                file_paths.append (os.path.join(path, name))
    return file_paths 


english_paths=getFiles(english)
greek_paths=getFiles(greek)
english_paths.sort()
greek_paths.sort()

english_and_greek_paths=[]

for i in range(len(english_paths)):
    english_and_greek_paths.append((english_paths[i],greek_paths[i]))





"""#VALIDATION set"""

import sys,os

#english_v="/Users/computer/PycharmProjects/Course_Work_Cornell_Tech2018/Deep_Learning/Font_Style_Models/pix2pix_juypter/val_characters/english"
#greek_v="/Users/computer/PycharmProjects/Course_Work_Cornell_Tech2018/Deep_Learning/Font_Style_Models/pix2pix_juypter/val_characters/greek"
english_v="MyDrive/deep_learning_characters_for_analysis/val_characters/english"
greek_v="MyDrive/deep_learning_characters_for_analysis/val_characters/greek"

english_file_paths_val=[]
greek_file_paths_val=[]

def getFiles(root):
    file_paths=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            if (".DS_Store" in name): pass 
            else:
                file_paths.append (os.path.join(path, name))
    return file_paths 


english_paths_val=getFiles(english_v)
greek_paths_val=getFiles(greek_v)
english_paths_val.sort()
greek_paths_val.sort()

english_and_greek_paths_v=[]

for i in range(len(english_paths_val)):
    english_and_greek_paths_v.append((english_paths_val[i],greek_paths_val[i]))



"""#CUda and img processing -n eed ro move locatio"""

netD.cuda()
netG.cuda()
inputA = inputA.cuda()
inputB = inputB.cuda()



#print((np.array(im)/255.0).shape)



def process_img(name_a):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        preprocessing = transforms.Compose([
            transforms.Resize((256,256)),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        image=io.imread(name_a)
        im = Image.fromarray(np.uint8(image))
        return preprocessing(im)

def read_img_simult(name_a,name_b):
    a=process_img(name_a) 
    b=process_img(name_b)
    
    return (a,b)

torch.cuda.empty_cache()

def get_all_img_pairings(paths):
  elements=[]
  for i in paths[0:]:
    print(i)
    elements.append(read_img_simult(i[0],i[1]))
  return elements

training_data=get_all_img_pairings(english_and_greek_paths)

val_data=get_all_img_pairings(english_and_greek_paths_v)

val_data=training_data[60:70]

def show_image(image):
    """Show image with landmarks"""
    #plt.savefig("a.png")
    plt.imshow(image)
    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001) 

#show_image(a.numpy().T)

len(training_data)

show_image(training_data[60:70][0][0].cpu().numpy().T)



"""#Mini batch encoding"""

#training_data=training_data+val_data
for_data_loader=Dataset_Handler(training_data)
for_val_loader=Dataset_Handler(val_data)

trainloader=torch.utils.data.DataLoader(for_data_loader, batch_size=20, shuffle=False, num_workers=0)
valloader=torch.utils.data.DataLoader(for_val_loader, batch_size=20, shuffle=False, num_workers=0)

dataiter = iter(trainloader)

len(dataiter.next()[0])



from IPython.display import display
def showX(X, rows=1):
    assert X.shape[0]%rows == 0
    int_X = ( (X+1)/2*255).clip(0,255).astype('uint8')
    if channel_first:
        int_X = np.moveaxis(int_X.reshape(-1,3,imageSize,imageSize), 1, 3)
    else:
        int_X = int_X.reshape(-1,imageSize,imageSize, 3)
    int_X = int_X.reshape(rows, -1, imageSize, imageSize,3).swapaxes(1,2).reshape(rows*imageSize,-1, 3)
    display(Image.fromarray(int_X))

channel_first=True
channel_axis=1
#train_batch = minibatch(english_and_greek_paths, 1, direction=direction)

#showX(trainA)
#showX(trainB)
#del train_batch, trainA, trainB

import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
trainA, trainB = next(dataiter)
plt.imshow(trainB[0].numpy().T)
plt.show()

optimizerD = optim.Adam(netD.parameters(), lr = lrD, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = lrG, betas=(0.5, 0.999))

loss = nn.BCELoss()
lossL1 = nn.L1Loss()
one = None
zero = None
def netD_train(A, B):    
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
    #print(errD_fake.data)
    #print(errD_real.data)
    return (errD_fake.data+errD_real.data)/2,


def netG_train(A, B):
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
    print( errG_L1.data)
    return errG_fake.data, errG_L1.data

def V(x):
    return Variable(x).cuda()

def netG_gen(A):
    return np.concatenate([netG(A[i:i+1]).data.cpu().numpy() for i in range(A.size()[0])], axis=0)

#_, valA, valB = next(val_batch)
#fakeB = netG_gen(V(valA))
#showX(np.concatenate([valA, valB, fakeB], axis=0), 3)

temp=iter(trainloader)

show_image(vA[0].cpu().numpy().T)
show_image(vB[0].cpu().numpy().T)
show_image(fakeB[0].T)

show_image(vA[0].cpu().numpy().T)
show_image(vB[0].cpu().numpy().T)
show_image(fakeB[0].T)

history_val=[] #so i can see how it develops over time
history_train=[] #so i can see how it develops over time 
history_fake=[] #BULLSHIT :(

import time
from IPython.display import clear_output
t0 = time.time()
niter = 30
gen_iterations = 0
errL1 = epoch = errG = 0
errL1_sum = errG_sum = errD_sum = 0

display_iters = 1
val_batch = iter(valloader)  #minibatch(english_and_greek_paths_v, 1, direction)
train_batch = iter(trainloader) #minibatch(english_and_greek_paths, 1, direction)
epoch=0

first_time=True
while epoch < niter: 
    trainA=None
    trainB=None
    try:
      trainA, trainB = next(train_batch)  
    except:
      epoch+=1
      trainloader=torch.utils.data.DataLoader(for_data_loader, batch_size=20, shuffle=True, num_workers=1)
      train_batch=iter(trainloader)
      trainA, trainB = next(train_batch)  
      first_time=True
      
    print(epoch)
    print(len(trainA))
    print(len(trainB))
    vA, vB = V(trainA), V(trainB)
    print(vA.shape)
    print(vB.shape)
    errD,  = netD_train(vA, vB)
    errD_sum +=errD

    # epoch, trainA, trainB = next(train_batch)
    errG, errL1 = netG_train(vA, vB)
    errG_sum += errG
    errL1_sum += errL1
    gen_iterations+=1
    
    if epoch%display_iters==0:
      if first_time==True:
        if gen_iterations%(5*display_iters)==0:
            clear_output()
        print('[%d/%d][%d] Loss_D: %f Loss_G: %f loss_L1: %f'
#         % (epoch, niter, gen_iterations, errD_sum/display_iters, 
           errG_sum/display_iters, errL1_sum/display_iters), time.time()-t0)
        channel_first=0
        
        #_, valA, valB = train_batch.send(6)
        
        vA, vB=vA, vB
        fakeB = netG_gen(V(vA))
        
        history_train.append(fakeB) #store history
       
        #show_image(valA[0].cpu().numpy().T)
        #show_image(valB[0].cpu().numpy().T)
        #show_image(fakeB[0].T)
        
        show_image(vA[0].cpu().numpy().T)
        show_image(vB[0].cpu().numpy().T)
        show_image(fakeB[0].T)
        #vA, vB = V(valA),V(valB)
        #fakeB = netG_gen(vA)
        #showX(np.concatenate([valA, valB, fakeB], axis=0), 3)
        first_time=False
        errL1_sum = errG_sum = errD_sum = 0
        val_batch = iter(valloader)
        valA, valB = next(val_batch)
        fakeB = netG_gen(V(valA))
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
        #showX(np.concatenate([vA.cpu(), vB.cpu(), fakeB], axis=0), 3)

val_batch = iter(trainloader)

#torch.save(netD.state_dict(), "des_D")
#torch.save(netG.state_dict(), "gen_D")
#valA, valB = next(val_batch)
#fakeB = netG_gen(V(valA))

for i in history_val: #p 
  show_image(i[0].T)



from google.colab import files

name="calibri_estimated_greek_later_training"
counter=1
for element in history_val:
  show_image(element[8].T)
 
  scipy.misc.imsave(name+"_"+str(counter)+"_x.jpg", ((element[8][2]+1)*255).astype(int))
  files.download(name+"_"+str(counter)+"_x.jpg") 
  counter+=1

for element_index in range(len(fakeB)):
  element=valA[element_index]
  show_image(element.cpu().numpy().T)
  
  
  element=valB[element_index]
  show_image(element.cpu().numpy().T)
 
  element=fakeB[element_index]
  show_image(element.T)
  
  
  print("NEXT")
  break
  
#fakeB = netG_gen(V(val_data[0]))

from google.colab import files

name="alt_estimated_greek_later_training"
counter=1
for element in fakeB:
  show_image(element.T)
 
  scipy.misc.imsave(name+"_"+str(counter)+".jpg", ((element[2]+1)*255).astype(int))
  files.download(name+"_"+str(counter)+".jpg") 
  counter+=1

name="alt_estimated_greek_later_training"
counter=1
for element in fakeB:
  show_image(element.T)
 
  scipy.misc.imsave(name+"_"+str(counter)+".jpg", ((element[2]+1)*255).astype(int))
  files.download(name+"_"+str(counter)+".jpg") 
  counter+=1

print (val_data[0][1].shape)

#!pip install pypng

current=temp[1]

#for i in range(len(current)):

temp=(list(netG[0].children())[0:2])

len(list(temp[1].children()))


outputs2= []
def hook2(module, input, output):
    #print(1)
    
    outputs2.append(output)

hk = list(list(temp[1].children())[1].children())[0].register_forward_hook(hook2)


#whuuut=temp[0](V(valA)).cpu().detach().numpy()

#whuuut
#show_image(temp[0](V(valA)).cpu().detach().numpy()[2][17])

#netG_gen(V(valA))

show_image(outputs2[0].cpu().detach().numpy()[0][39])
#outputs2[0].shape

scipy.misc.imsave('outfile.jpg', ((element[2]+1)*255).astype(int))

"""-------------------------"""
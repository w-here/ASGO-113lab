import torch

import torchvision
from torch.utils.data import DataLoader,Dataset
import copy as cp
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class myData(Dataset):
    def __init__(self,dataPath,transform=None):
        super(myData,self).__init__()
        # data = pd.read_csv(dataPath)
        self.data = pd.read_csv(dataPath)
        self.transform = transform


    def __getitem__(self,index):
        #img1 = self.data.loc[index,'img1']
        # img2 = self.data.loc[index,'img2']
        # gt = self.data.loc[index,'GT']
        transform = self.transform
        #img1 = mpimg.imread(self.data.loc[index,'img1'])[:,:,:3]
        img1 = mpimg.imread(self.data.loc[index,'img1'])
        #print(img1.shape)
        #img1=nn.Upsample(size=(256,256,3),mode='nearest')
        #print(img1.size)
        img2 = mpimg.imread(self.data.loc[index,'img2'])[:,:,:3]
        #img2=img2.reshape((256,256,3))
        gt = mpimg.imread(self.data.loc[index,'GT']) # batch_size*1*112*112
        gt_name = self.data.loc[index,'GT']

        gt = gt.reshape((256,256,-1))[:,:,0:1]
        # s =cp.deepcopy( gt)
        #gt[gt > 40] = 255
        #gt[gt <= 40] = 0
        #print(sss)
        #gt = np.concatenate((s,1-s),axis=2)
       # gt=torch.tensor(gt)
       #  print(type(gt))
       #  print(gt.shape)
       #  print(gt)
      #  print(gt.shape)
        #gt = gt.reshape((112,112,2))
        #print("Hello3",gt.shape)
        if transform:
            img1 = transform(img1)
            #print(img1.shape)
            img2 = transform(img2)
            gt = transform(gt)

        return img1,img2,gt,gt_name
    
    def __len__(self):

        return len(self.data)

if __name__=="__main":
    mydata = myData()
    img1,img2,gt = mydata.__getitem__(2)
    print(gt.shape)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models_changeNet import *
import torchvision
from torchvision import datasets, models, transforms
#from Unetpp import *

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#from ASPP_dif_Unet import *
#from UP_ASPP_dif_Unet import *
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
#from tensorboardX import SummaryWriter
#from Deepunet import *
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#from unet import *
import os
#from rsnetbasicbam import *

#import  hrnetbasicdifatt as rs
#import  hrnetatt as rs
import ADHR as adhr

#from rsnet import *
#from dif_Unet import *
import logging
#import cv2
from math import ceil, sqrt
#from siamunet_diff import *
from test_configs import Config
#from models import *
#from DualUnet import*
#from DualUnet_UP import *
from data_loader import myData
#from utils import log,myshow
#from difuntaspp import *
#from difuntaspp2 import *
#from difuntDU import *
#from Unetpp import *
# %%
model_list = []
for root,dirs,files in os.walk('/home/C/tm/model/'):
    for file in files:
        model_list.append(root+file)
for idx,item in enumerate(model_list):
    print("%d:%s"%(idx,item))


# %%
cf = Config()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%

#net=dif_UNet().to(device)
#net=NestedUNet().to(device)
#net=DualUNet().to(device)
#net=dif_UNet_UP().to(device)
net=adhr.ADHR().to(device)
#net=RSnet().to(device)
#net=rs.RSnet_1().to(device)
#net=rs.BASE_Transformer().to(device)
#net=dif_UNet_DU().to(device)
#net=rs.SiameseNet().to(device)
#net=dif_UNet_ASPP().to(device)
#net=ASPP_dif_UNet().to(device)
#net=UP_ASPP_dif_UNet().to(device)
#net=deepunet(6,2)

for i in range(len(model_list)):

  f = open('result.txt', 'a')
  net.load_state_dict(torch.load(model_list[i]))
  #print(i)
  #print(model_list[i])
  f.write(model_list[i]+"\n")

#

# %%
  data_transforms = transforms.Compose([
      transforms.ToTensor(),
  ])

#testL1
  #testgoo37.csv
  #testgoo

  #valw.csv whutest7000.scv
  #valL.csv  testL1.csv
  #/ valL.csv valw2000.csv
  testSet = myData(dataPath='./test.csv',
                 transform=data_transforms)

  testLoader = DataLoader(dataset=testSet,
                        batch_size=cf.batch_size,
                        shuffle=False,
                        num_workers=cf.num_workers)

  test_out=list()
  out_save0=list()
  out_save1=list()
  out_save2=list()
  out_save3=list()

  out_save=list()
  test_gt=list()
  gt_save=list()

  with torch.no_grad():
      for batch, data in enumerate(testLoader):
          img1, img2, gt, gt_name = data[0].to(device), data[1].to(device), data[2].to(device), data[3]
    # img1=img1.type(torch.FloatTensor)
    #
    # img2 = img2.type(torch.FloatTensor)
#with torch.no_grad():
        #o1,o2,output = net(img1,img2)
          net.eval()
          output = net(img1, img2)

          output_final=output
    ########################################
    #output_final0=output[0]
    # output_final1=output[1]
    # output_final2=output[2]
    # output_final3=output[3]
    ########################################

          final = output_final.cpu().detach().numpy()
          final[final >= 0.50] = 1.0
          final[final < 0.50] = 0.0

          locs = torch.from_numpy(final)

          gt[gt <0.01] = 0
          gt[gt >=0.01] = 1
   ############################
          out_save.append(locs)
          gt_save.append(gt)
    ###########################
          gt_y=gt.cpu().numpy()
          out_y=locs.cpu().numpy()
          test_gt.append(gt_y)
          test_out.append(out_y)


    ###################################
   # out_save0.append(output_final0)
    # out_save1.append(output_final1)
    # out_save2.append(output_final2)
    # out_save3.append(output_final3)
    ###################################

  TP_save=list()
  TN_save=list()
  FP_save=list()
  FN_save=list()


  TP,TN,FP,FN,N = 0,0,0,0,0
  for item in range(len(test_out)):
      TN=np.sum((1-test_out[item])*(1-test_gt[item]))
      TP=((test_gt[item])*(test_out[item])).sum()
      FN=(test_gt[item]*(1-test_out[item])).sum()
      FP=((1-test_gt[item])*(test_out[item])).sum()

      FN_save.append(FN)
      TP_save.append(TP)
      TN_save.append(TN)
      FP_save.append(FP)

  N=256*256*2


  TP_ALL=sum(TP_save)
  TN_ALL=sum(TN_save)
  FP_ALL=sum(FP_save)
  FN_ALL=sum(FN_save)

  #print(len(test_out))
  f.write(str(len(test_out))+"\n")
  N_ALL=N*(len(test_out))
  #print("TP",TP_ALL)
  f.write("TP:"+str(TP_ALL)+"\n")
  #print("FP",FP_ALL)
  f.write("FP:"+str(FP_ALL)+"\n")
  #print("TN",TN_ALL)
  f.write("TN:"+str(TN_ALL)+"\n")
  #print("FN",FN_ALL)
  f.write("FN:"+str(FN_ALL)+"\n")
  #print("N",N_ALL)
  f.write("N:"+str(N_ALL)+"\n")

  ACC=(TP_ALL+TN_ALL)/N_ALL
  PRE=((TP_ALL+FP_ALL)*(TP_ALL+FN_ALL)+(FN_ALL+TN_ALL)*(FP_ALL+TN_ALL))/(N_ALL*N_ALL)
  kappa=(ACC-PRE)/(1-PRE)
  Pr=TP_ALL/(TP_ALL+FP_ALL)
  Re=TP_ALL/(TP_ALL+FN_ALL)
  F1=2*Pr*Re/(Pr+Re)
  #print("PCC:",ACC)
  f.write("PCC:"+str(ACC)+"\n")
  #print("Pr:",Pr)
  f.write("Pr:"+str(Pr)+"\n")
  #print("Re:",Re)
  f.write("Re:"+str(Re)+"\n")
  #print("F1:",F1)
  f.write("F1:"+str(F1)+"\n")
  #print("kappa",kappa)
  f.write("Kappa:"+str(kappa)+"\n")
  f.write("\n")
  f.close()




#save image
#test_root = './saveL/'  1
#for item in range(len(test_out)):  2
 #   a = out_save[item]  3
  #  b = gt_save[item]  4
    #c = out_save0[item]
    # d = out_save1[item]
    # e=  out_save2[item]
    # f= out_save3[item]

    # torchvision.utils.save_image(c, test_root + str(item) + '-out0.png', padding=0)
    # torchvision.utils.save_image(d, test_root + str(item) + '-out1.png', padding=0)
    # torchvision.utils.save_image(e, test_root + str(item) + '-out2.png', padding=0)
    # torchvision.utils.save_image(f, test_root + str(item) + '-out3.png', padding=0)

   # torchvision.utils.save_image(a, test_root + str(item) + '-out.png', padding=0)  5
   # torchvision.utils.save_image(b,test_root+str(item)+'-gt.png',padding=0)  6


import torchvision.models as models
import torch
import copy
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):# Net for region based segment classification
######################Load main net (resnet 50) class############################################################################################################
        def __init__(self): # Load pretrained encoder and prepare net layers
            super(Net, self).__init__()
# ---------------Load pretrained torchvision resnet (need to be connected to the internet to download model in the first time)----------------------------------------------------------
            self.Net = models.resnet101(pretrained=True)
#----------------Change final layer to predict 3 class as binary yes/no for each class------------------------------------------------------------------------------------------
            self.Net.fc=nn.Sequential(nn.Linear(2048, 1024),nn.ReLU())
            self.LinkLayer=nn.Linear(1024,2)
            self.ContainLayer=nn.Linear(1024, 2)
            self.InsideLayer=nn.Linear(1024, 2)
#----------------------Add attention layer to proccess the vessel masks------------------------------------------------
            self.AttentionLayerMask1 = nn.Conv2d(1, 64, stride=2, kernel_size=3, padding=1, bias=True)
            self.AttentionLayerMask2 = nn.Conv2d(1, 64, stride=2, kernel_size=3, padding=1, bias=True)




###############################################Run prediction inference using the net ###########################################################################################################
        def forward(self,Images, Mask1, Mask2,TrainMode=True,UseGPU = True):

#------------------------------- Convert from numpy to pytorch-------------------------------------------------------
                if TrainMode:
                    mode=torch.FloatTensor
                else:
                    mode=torch.half


                mode = torch.FloatTensor


                self.type(mode)
                InpImages = torch.autograd.Variable(torch.from_numpy(Images), requires_grad=False).transpose(2,3).transpose(1, 2).type(mode)
                ROIMask1 = torch.autograd.Variable(torch.from_numpy(Mask1.astype(np.float)), requires_grad=False).unsqueeze(dim=1).type(mode)
                ROIMask2 = torch.autograd.Variable(torch.from_numpy(Mask2.astype(np.float)), requires_grad=False).unsqueeze(dim=1).type(mode)
                self.Net = self.Net.type(mode)
                if UseGPU == True: # Convert to GPU
                    InpImages = InpImages.cuda()
                    ROIMask1 = ROIMask1.cuda()
                    ROIMask2 = ROIMask2.cuda()
                    self.cuda()
                else:
                    self.cpu()
# -------------------------Normalize image-------------------------------------------------------------------------------------------------------
                RGBMean = [123.68, 116.779, 103.939]
                RGBStd = [65, 65, 65]
                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # Normalize image by std and mean

#============================Run net layers===================================================================================================

                x=InpImages
                #----------------First resnet layer--------------------------------------------------------------------------------------------------
                x = self.Net.conv1(x) # First resnet convulotion layer
                x = self.Net.bn1(x)
                x = self.Net.relu(x)
                # ----------------Apply Attention layer--------------------------------------------------------------------------------------------------
                x = x + self.AttentionLayerMask1(ROIMask1)+self.AttentionLayerMask2(ROIMask2) # Procces vessel masks and add to main stream
                # ---------------------First resnet block-----------------------------------------------------------------------------------------------

                x = self.Net.maxpool(x)
                x = self.Net.layer1(x)
                # --------------------Second Resnet  Block------------------------------------------------------------------------------------------------
                x = self.Net.layer2(x)
                x = self.Net.layer3(x)
                # -----------------Resnet  block 4---------------------------------------------------------------------------------------------------
                x = self.Net.layer4(x)

                # ------------Fully connected final vector--------------------------------------------------------------------------------------------------------
                x = torch.mean(torch.mean(x, dim=2), dim=2)
                #x = x.squeeze()
                x = self.Net.fc(x)
                #---------------------Final Predictions layer for each class------------------------------------------------------------------------------------------------------


                Link=self.LinkLayer(x)
                Contain=self.ContainLayer(x)
                Inside=self.InsideLayer(x)

                #_____________________SoftMax for each class predict yes/no___________________________________
                ProbDic={}
                ProbDic["Linked"] = F.softmax(Link,dim=1) # Probability vector
                ProbDic["Inside"] = F.softmax(Contain,dim=1) # Probability vector
                ProbDic["Contain"] = F.softmax(Inside,dim=1) # Probability vector


               # Prob,PredLink.max(dim=1) # Top predicted class and probability


                return  ProbDic
###################################################################################################################################
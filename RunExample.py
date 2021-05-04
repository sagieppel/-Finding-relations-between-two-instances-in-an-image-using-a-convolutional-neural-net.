# Run example on single image and two vessel masks (should run out of the box for a the example image and masks)
#...............................Imports..................................................................
import numpy as np
import torch
import Visuallization as vis
import Net as NetBuild
import os
import cv2
##################################Input paramaters#########################################################################################


TestImage="ExamplePredict/Image.jpg" # Input image
TestMask1="ExamplePredict/Mask1.png" # input vessel mask
TestMask2="ExamplePredict/Mask2.png" # input vessel mask
OutDir="OutPrediction/"
if not os.path.exists(OutDir): os.mkdir(OutDir)
Trained_model_path="logs/Defult.torch" # Pretrain model path
UseGPU=False
#---------------------Create and Initiate net ------------------------------------------------------------------------------------
Net=NetBuild.Net() # Create net and load pretrained
Net.load_state_dict(torch.load(Trained_model_path))
#------------------------Load images and masks----------------------------------------------------
Img=cv2.imread(TestImage)
Msk1=cv2.imread(TestMask1,0)
Msk2=cv2.imread(TestMask2,0)
#------------------Pre procces image and masks-----------------------------------------------------
Img=np.expand_dims(Img,axis=0)
Msk1=np.expand_dims(Msk1,axis=0)>0
Msk2=np.expand_dims(Msk2,axis=0)>0
#--------------------Run prediction--------------------------------------------------------------
with torch.no_grad():
    PredDic = Net.forward(Img, Msk1, Msk2, TrainMode=False, UseGPU=UseGPU)  # run prediction
#----------------Display Save Predictions------------------------------------------------
PRedTxt=""
for cls in PredDic:
       if (PredDic[cls][0,1]>0.5):
           PRedTxt+=cls+"_"
print("Relation Predicted:"+PRedTxt)
I=Img[0].copy()
I[:,:,0][Msk1[0]>0]=0
I[:,:,1][Msk1[0]>0]=255
#
I[:,:,1][Msk2[0]>0]=0
I[:,:,0][Msk2[0]>0]=255

VisImg=np.concatenate([Img[0],I],axis=1)
cv2.imwrite(OutDir+"/"+PRedTxt+".jpg",VisImg)
vis.show(VisImg ,"Relation Predicted:"+PRedTxt)








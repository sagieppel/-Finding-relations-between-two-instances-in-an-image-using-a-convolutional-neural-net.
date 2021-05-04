# Evaluate relation prediction model on evaluation set
#...............................Imports..................................................................
import os
import numpy as np
import torch
import Reader as LabPicsReader

import Net as NetBuild

import Visuallization as vis
##################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................

EvaluateInputDir="TrainExample/" # Input data to evaluate

Trained_model_path="logs/Defult.torch" # input model weight to evaluate

UseGPU=False # Use GPU for prediction
#---------------------Create and Initiate net ------------------------------------------------------------------------------------
Net=NetBuild.Net() # Create net and load pretrained
Net.load_state_dict(torch.load(Trained_model_path))
#-----------------Create data reader---------------------------------------------------------------------------------

Reader=LabPicsReader.Reader(EvaluateInputDir)



#..............Statitics....................................................................
ClassList=["Linked","Inside","Contain","None"]
TP={} # True positves sum
FP={} # False positives sum
FN={} # False negative sum
for cls in ClassList:
    TP[cls]=FP[cls]=FN[cls]=0
#................Main evaluation loop....................................................................
while(Reader.epoch==0):
    Img, Msk1, Msk2, GTClass = Reader.LoadSingleClean(MaxImageSize=1000000) # Read next batch (single image, vessel mask pair and label)
#***********************************************************************************************
    # for i in range(Img.shape[0]):
    #     Image = Img[i]
    #     Image[:, :, 0][Msk1[i] > 0] = 0
    #     Image[:, :, 1][Msk2[i] > 0] = 0
    #     txt = ""
    #     for cls in GTClass:
    #         if GTClass[cls][i]: txt += cls + "   "
    #     print(txt)
    #     vis.show(Image, txt)
#*****************************************************************************************************
    with torch.no_grad():
        PredDic = Net.forward(Img, Msk1, Msk2,TrainMode=False,UseGPU = UseGPU) # run prediction

#.............Calculate satatistics
    NonePrd=True # no relationship predicted
    NoneGT=True # no relationship GT
    for cls in PredDic:
       PrdCl=(PredDic[cls][0,1]>0.5)
       GTCl=GTClass[cls][0]>0.5

       if PrdCl:
           NonePrd=False
           if GTCl:
               NoneGT=False
               TP[cls]+=1
           else:
               FP[cls]+=1
       else:
           if GTCl:
               NoneGT=False
               FN[cls]+=1

    if NoneGT:
        if NonePrd:
            if NoneGT:
                TP["None"]+=1
            else:
                FP["None"]+=1
        else:
            if NoneGT:
                FN["None"]+=1


#..........................Calculate Final statics and display......................................................................
    for cls in ClassList:
        Precision=TP[cls]/(TP[cls]+FP[cls]+0.00001)
        Recall=TP[cls]/(TP[cls]+FN[cls]+0.00001)
        IOU=TP[cls]/(TP[cls]+FN[cls]+FP[cls]+0.00001)
        print(cls+"\tPrecision\t"+str(Precision)+"\tRecall\t"+str(Recall)+"\tIOU\t"+str(IOU)+"\tNUM CASES\t"+str(TP[cls]+FN[cls]))




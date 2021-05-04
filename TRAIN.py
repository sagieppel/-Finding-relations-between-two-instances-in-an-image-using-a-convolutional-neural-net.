# Train net to predict relations between two vessel masks in an image
#...............................Imports..................................................................
import os
import numpy as np
import torch
import Reader as LabPicsReader
import Net as NetBuild
import Visuallization as vis
##################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................
ChemLabPicsDir="TrainExample/" # Path to training annotation LabPics chemistry
MedLabPicsDir="TrainExample/"  # Path to training annotation LabPics medical
Learning_Rate=1e-5 # learning rate
BatchSize=2 # Number of images per batch
#************** Set folder for trained model**************************************************************************************************************************************
TrainedModelWeightDir="logs/" # Folder where trained model weight and information will be stored"
if not os.path.exists(TrainedModelWeightDir):
    os.mkdir(TrainedModelWeightDir)
Trained_model_path="" # Path of trained model weights If you want to return to trained model, else should be =""

#=========================Load net weights if exist====================================================================================================================
InitStep=1
if os.path.exists(TrainedModelWeightDir + "/Defult.torch"):
    Trained_model_path=TrainedModelWeightDir + "/Defult.torch"
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate.npy"):
    Learning_Rate=np.load(TrainedModelWeightDir+"/Learning_Rate.npy")
if os.path.exists(TrainedModelWeightDir+"/itr.npy"): InitStep=int(np.load(TrainedModelWeightDir+"/itr.npy"))
#...............Other training paramters..............................................................................

TrainLossTxtFile=TrainedModelWeightDir+"TrainLoss.txt" #Where train losses will be writen
Weight_Decay=1e-5# Weight for the weight decay loss function
MAX_ITERATION = int(100000010) # Max  number of training iteration
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NetBuild.Net() # Create net and load pretrained


if Trained_model_path!="": # Optional initiate full net by loading a pretrained net weights
    Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.cuda() # Use GPU


optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) # Create adam optimizer


#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------

ChemReader=LabPicsReader.Reader(ChemLabPicsDir)
MedReader=LabPicsReader.Reader(MedLabPicsDir)

#--------------------------- Create logs files for saving loss statistics during training----------------------------------------------------------------------------------------------------------
if not os.path.exists(TrainedModelWeightDir): os.makedirs(TrainedModelWeightDir) # Create folder for trained weight
f = open(TrainLossTxtFile, "w+")# Training loss log file
f.write("Iteration\tloss\t Learning Rate=")
f.close()
AVGLoss=0 # running average loss
#..............Start Training loop: Main Training....................................................................
print("Start Training")
for itr in range(InitStep,MAX_ITERATION): # Main training loop

#----------------------Load next batch---------------------------------------------------------------------
    if np.random.rand()<0.6:
                Img, Msk1, Msk2, GTClass = ChemReader.LoadRandomAugmentedBatch(Bsize=BatchSize) # Load From lab pics chemistry
    else:
                Img, Msk1, Msk2, GTClass = MedReader.LoadRandomAugmentedBatch(Bsize=BatchSize) # Load from labpics medical
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

    PredDic = Net.forward(Img, Msk1, Msk2,TrainMode=True,UseGPU = True) # Run prdiction for predicting relation between msk1 and msk2 in Image img
    Net.zero_grad()
    Loss = 0
#------------Calculate loss for each catagory-----------------------------------------------------------------------------------------------
    for cls in PredDic:
       GTcls=torch.autograd.Variable(torch.from_numpy(GTClass[cls]).cuda(), requires_grad=False)
       Loss += -torch.mean((GTcls * torch.log(PredDic[cls][:,1] + 0.0000001)))  # Calculate cross entropy loss
       Loss += -torch.mean(((1-GTcls) * torch.log(PredDic[cls][:, 0] + 0.0000001)))  # Calculate cross entropy loss
#-----------------------backpropogate----------------------------------------------------------------------------------------------------------------
    Loss.backward()  # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight
    torch.cuda.empty_cache()
    #continue
######Calculate running average loss############################################################
    fr = 1 / np.min([itr - InitStep + 1, 2000])
    AVGLoss = AVGLoss * (1-fr) + fr * float(Loss.data.cpu().numpy()) # Average loss
#=================== save statitics and displaye loss======================================================================================
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 1000 == 0 and itr>0: #Save model weight and other paramters in temp file once every 1000 steps
        print("Saving Model to file in "+TrainedModelWeightDir+"/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/DefultBack.torch")
        print("model saved")
        np.save(TrainedModelWeightDir+"/Learning_Rate.npy",Learning_Rate)
        np.save(TrainedModelWeightDir+"/itr.npy",itr)
    if itr % 30000 == 0 and itr>0: #Save model weight once every 30k steps permenant (not temp)
        print("Saving Model to file in "+TrainedModelWeightDir+"/"+ str(itr) + ".torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")
#......................Write and display train loss..........................................................................
    if itr % 10==0: # Display  and save train loss

        txt="\n"+str(itr)+"\t Average Loss "+str(AVGLoss) +"\n"

        print(txt)
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write(txt)
            f.close()



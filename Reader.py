## Reader Labpics dataset vessel relations



import numpy as np
import os
import cv2
import json
import threading
import Visuallization as vis

##############################################################################################
ClassList=["Linked","Inside","Contain","None"]
#########################################################################################################################
class Reader:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, MainDir=r""):
        self.epoch = 0 # Training Epoch
        self.itr = 0 # Training iteratation
# ----------------------------------------Create list of annotations arranged by class--------------------------------------------------------------------------------------------------------------
        self.AnnList = [] # Image/annotation list
        self.AnnByType = {} # Image/annotation list by class
        for cls in ClassList:
           self.AnnByType[cls] = [] # Annotation divided by type inside/contain/link/none
        print("Creating annotation list for reader this might take a while")
        for AnnDir in os.listdir(MainDir): #  Read all vessel pairs in images and find their relations
            ImageDir=MainDir + "/" + AnnDir
            data = json.load(open(ImageDir + '/Data.json', 'r')) # Read annotation data



            if len(data['Vessels'])==1: continue
            for vesName in data['Vessels']: # check relation between every two vessels pairs in the image
                vdat=data['Vessels'][vesName]
                vind=vdat["Indx"]
                for vesName2 in data['Vessels']:
                    Entry = {} # Data on specific vessel pairs
                    Entry["Image"] = ImageDir + "/Image.jpg"
                    Entry["Mask1"] = ImageDir + "//Vessels//" + str(vind) + ".png"
                    if vesName==vesName2: continue
                    vdat2 = data['Vessels'][vesName2]
                    vind2 = vdat2["Indx"]
                    Entry["Mask2"] = ImageDir + "//Vessels//" + str(vind2) + ".png"
                    Entry["Linked"] = vind2 in vdat['LinkedToVessels_Indx'] # Check link relation between vessels
                    Entry["Inside"] = vind2 in vdat['InsideOfVessels_Indx'] # Check inside relation between vessels
                    Entry["Contain"] = vind2 in vdat['ContainVessels_Indx']
                    Entry["None"] = not (Entry["Linked"] or  Entry["Inside"] or Entry["Contain"])
                    self.AnnList.append(Entry) # Arrays of all vessel pairs
                    for cls in ClassList:
                        if Entry[cls]: self.AnnByType[cls].append(Entry) # Vessel pairs by class


######################################################Augmented Image add noise##################################################################################################################################
    def Augment(self,Img,Mask1,Mask2):
        Img=Img.astype(np.float)
        if np.random.rand()<0.5: # flip left right
            Img = np.fliplr(Img)
            Mask1 = np.fliplr(Mask1)
            Mask2 = np.fliplr(Mask2)


        # if np.random.rand()<0.0: # flip  up down
        #             Img = np.flipud(Img)
        #             Mask1 = np.fliplr(Mask1)
        #             Mask2 = np.fliplr(Mask2)
        if np.random.rand()<0.2: # Change from rgb to bgr
            Img = Img[..., :: -1]

        if np.random.rand() < 0.2: # Add noise
            noise = np.random.rand(Img.shape[0],Img.shape[1],Img.shape[2])*0.2+np.ones(Img.shape)*0.9
            Img *=noise
            Img[Img>255]=255

        if np.random.rand() < 0.2: # Gaussian blur
            Img = cv2.GaussianBlur(Img, (5, 5), 0)

        if np.random.rand() < 0.5:  # Dark light
            Img = Img * (0.5 + np.random.rand() * 0.65)
            Img[Img>255]=255

        if np.random.rand() < 0.2:  # GreyScale
            Gr=Img.mean(axis=2)
            r=np.random.rand()

            Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
            Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
            Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)


        return Img, Mask1, Mask2
#############################################Load batch for training #####################################################################################################################
    def LoadRandomAugmentedBatch(self,Bsize,Hbc=0,Wbc=0,MaxImageSize=1000000):
        # ------------------------------------------------------------------------------------------------------------
        for i in range(Bsize):
            Class = list(self.AnnByType)[np.random.randint(len(list(self.AnnByType)))] # Pick random class for batch inside/contain/link
            while (True):
                #..................Loac vessel masks and image..................................................................
                Entry = self.AnnByType[Class][np.random.randint(len(self.AnnByType[Class]))]
                msk1 = cv2.imread(Entry["Mask1"], 0) > 0
                msk2 = cv2.imread(Entry["Mask2"], 0) > 0
                img = cv2.imread(Entry["Image"])

                ###############Crop Image and Mask##############################################################################33
                h, w = msk1.shape
                bbox = cv2.boundingRect((msk1 + msk2).astype(np.uint8))
                Xmn = bbox[0]
                Ymn = bbox[1]
                Wbx = bbox[2]
                Hbx = bbox[3]
                Xmx = Xmn + Wbx
                Ymx = Ymn + Hbx
                y0 = x0 = 0
                x1 = w
                y1 = h
                if Ymn > 0: y0 = np.random.randint(Ymn)
                if Xmn > 0: x0 = np.random.randint(Xmn)
                if h > Ymx: y1 = np.random.randint(Ymx, h)
                if w > Xmx: x1 = np.random.randint(Xmx, w)

                img = img[y0:y1, x0:x1]
                msk1 = msk1[y0:y1, x0:x1]
                msk2 = msk2[y0:y1, x0:x1]
                #####################Create batch###############################################################333
                if i==0:
                    if Hbc <= 0: Hbc=img.shape[0]
                    if Wbc <= 0: Wbc=img.shape[1]
                    if np.random.rand()<0.3:
                        R = np.random.rand()+0.5
                        Hbc = int(R * Hbc)
                        Wbc = int(R * Wbc)
                    if Wbc*Hbc>MaxImageSize: # if too big shrink
                        R=(MaxImageSize/Wbc/Hbc)**0.5
                        Hbc=int(R*Hbc)
                        Wbc=int(R*Wbc)
                    if np.min([Wbc,Hbc])<100: # if to small resize
                        R = 100/np.min([Wbc,Hbc])
                        Hbc = int(R * Hbc)
                        Wbc = int(R * Wbc)

                    Bmsk1 = np.zeros([Bsize,Hbc,Wbc],np.float32)
                    Bmsk2 = np.zeros([Bsize, Hbc, Wbc], np.float32)
                    BImg =  np.zeros([Bsize, Hbc, Wbc, 3], np.float32)

                    Bclass = {}
                    for cls in ClassList:
                        Bclass[cls] = np.zeros([Bsize], np.float32)
                elif (Hbc/Wbc)>(msk1.shape[0]/msk1.shape[1])*3 or (Hbc/Wbc)*3<(msk1.shape[0]/msk1.shape[1]): continue
                #---------------------------Augment----------------------------------------------
                img,msk1,msk2 = self.Augment(img,msk1,msk2)

                #------------------------Put data into training batch------------------------------------------------
                Bmsk1[i,:,:] = cv2.resize(msk1.astype(np.uint8),(Wbc,Hbc), interpolation=cv2.INTER_NEAREST)
                Bmsk2[i,:,:] =cv2.resize(msk2.astype(np.uint8),(Wbc,Hbc), interpolation=cv2.INTER_NEAREST)
                BImg[i] = cv2.resize(img.astype(np.uint8),(Wbc,Hbc), interpolation=cv2.INTER_LINEAR)

                for cls in ClassList:
                     if Entry[cls]: Bclass[cls][i]=1
                break

        return BImg, Bmsk1, Bmsk2, Bclass

  ##############################################################################################################

  #                         Load data for a single pair of vessels sequentially and with no augmantation for evaluation

  ########################################################################################################
    def LoadSingleClean(self,MaxImageSize=1000000): # For Evaluation
        self.itr += 1
        print(self.itr)
        if self.itr >= len(self.AnnList):
            self.itr = 0
            self.epoch += 1
#.............Load next pair entry.........................................................................

        #...................Load image and vessel masks..............................................................
        Entry = self.AnnList[self.itr]
        msk1 = cv2.imread(Entry["Mask1"], 0) > 0
        msk2 = cv2.imread(Entry["Mask2"], 0) > 0
        img = cv2.imread(Entry["Image"])

        #....................Resize if to big or small...........................................................
        Hbc = img.shape[0]
        Wbc = img.shape[1]
        if np.random.rand() < 0.3:
            R = np.random.rand() + 0.5
            Hbc = int(R * Hbc)
            Wbc = int(R * Wbc)
        if Wbc * Hbc > MaxImageSize:  # if too big shrink
            R = (MaxImageSize / Wbc / Hbc) ** 0.5
            Hbc = int(R * Hbc)
            Wbc = int(R * Wbc)
        if np.min([Wbc, Hbc]) < 100:  # if to small resize
            R = 100 / np.min([Wbc, Hbc])
            Hbc = int(R * Hbc)
            Wbc = int(R * Wbc)
#...............Create batch.................................................................
        Bmsk1 = np.zeros([1, Hbc, Wbc], np.float32)
        Bmsk2 = np.zeros([1, Hbc, Wbc], np.float32)
        BImg = np.zeros([1, Hbc, Wbc, 3], np.float32)

        Bclass = {}
        for cls in ClassList:
            Bclass[cls] = np.zeros([1], np.float32)
#........................Load data into batch..............................................................
        Bmsk1[0, :, :] = cv2.resize(msk1.astype(np.uint8), (Wbc, Hbc), interpolation=cv2.INTER_NEAREST)
        Bmsk2[0, :, :] = cv2.resize(msk2.astype(np.uint8), (Wbc, Hbc), interpolation=cv2.INTER_NEAREST)
        BImg[0] = cv2.resize(img.astype(np.uint8), (Wbc, Hbc), interpolation=cv2.INTER_LINEAR)

        for cls in ClassList: # Load relations into batch classes
            if Entry[cls]: Bclass[cls][0] = 1
        return BImg, Bmsk1, Bmsk2, Bclass
###########################Load Image to batch

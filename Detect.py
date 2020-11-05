from __future__ import print_function   #Use newest way to print if has new version in future
from __future__ import division         #Use newest way to division if has new version in future

from time import sleep  #import sleep lib as delay in Microcontroller



#import serial           #import Serial(UART) lib , need enable hardware uart in Rasp's setting
import os
import numpy as np
import cv2              #import opencv lib

import time

cam = cv2.VideoCapture(0)

cv2.namedWindow("input_money")

###############DETECT CODE###########################################################################

lower1 = np.array([81,35,141])
upper1 = np.array([157,255,255])

lower2 = np.array([49,113,70])
upper2 = np.array([206,255,109])

Ratio = 0.90            # Rate of distance, greater will be checked easier



# With surf and sift we can use bf or flann, akaze only use akaze
#detector=cv2.xfeatures2d.SIFT_create()
#detector = cv2.xfeatures2d.SURF_create()
detector = cv2.AKAZE_create()

#FLANN
FLANN_INDEX_KDITREE=0   #Procedures
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)   #Procedures
flann=cv2.FlannBasedMatcher(flannParam,{})  #Procedures

#BF
#BF = cv2.BFMatcher()

#AKAZE
AKAZE = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

# This is an array, each of the elements is a name directory of image.
# Dataset array
TraingIMGArr = ["TrainingData/10000F.png","TrainingData/10000B.png",
                "TrainingData/20000F.png","TrainingData/20000B.png",
                "TrainingData/50000F.png","TrainingData/50000B.png",
                "TrainingData/100000F.png","TrainingData/100000B.png",
                "TrainingData/200000F.jpg","TrainingData/200000B.jpg",
                "TrainingData/500000F.png","TrainingData/500000B.png"
                ]

# Use to print to console and LCD
PrintingElement = ["10K","10K",
                    "20K","20K",
                    "50K","50K",
                    "100K","100K",
                    "200K","200K",
                    "500K","500K"
                    ]


#Loading features of dataset to DesArr 
DesArr = np.load("feature.npy",allow_pickle=True) 


while(1):     
    ret, frame = cam.read()
    cv2.imshow("input_money", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Thoat chuong trinh!!")
        break
    if k%256 == 32:
        # SPACE pressed
        img_name = "img_input.png"
        cv2.imwrite(img_name, frame)
        print("That is:")
         # Read image has just taken
        Raw_usr_img=cv2.imread("img_input.png") #Read img from user (captured from raspberry)
        PhotoDetect = cv2.resize(Raw_usr_img, (640,480))
        PhotoDetect=PhotoDetect[130:420,40:630]
        hsv_img = cv2.cvtColor(PhotoDetect, cv2.COLOR_BGR2HSV)   # HSV image
        
        mask_sub1 = cv2.inRange(hsv_img , lower1, upper1)
        mask_sub2 = cv2.inRange(hsv_img , lower2, upper2)
        mask = cv2.bitwise_or(mask_sub1,mask_sub2)
        _,contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        suma=0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            suma=suma+area
        result=suma/(PhotoDetect.shape[0]*PhotoDetect.shape[1])*100
        #print(result)
        if result>120:
            print ("PHOTO MONEY")
            
        else:
            Raw_usr_img=cv2.imread("img_input.png") #Read img from user (captured from raspberry)
            queryKP,queryDesc=detector.detectAndCompute(Raw_usr_img,None) #Procedures to get feature from this picture
        
            max_point = 0; # Max point
            index_element_arr = 0; # Index which picture are detecting or detected to print out LCD or console

          

            for i in range(len(TraingIMGArr)):            
                matches=AKAZE.knnMatch(queryDesc,DesArr[i],k=2) #Procedures 

                #print("DETECTING - " + PrintingElement[i]) #Print to console which image are being processed
           

                Match_Count = 0 # Create a variable to count match points from 2 images
                for m,n in matches:
                    if(m.distance < Ratio * n.distance):   #If match 
                        Match_Count += 1    #increase by 1
                #print(Match_Count)  #Print to console, comment if don't need it
                if Match_Count >= max_point: # If the Match_Count greater than max_point
                    max_point = Match_Count  # Assign max_point again
                    index_element_arr = i;   # Assign idex to print to console and LCD 
            # Get end time
            end = time.time()
            #print(end - start)
            
            #If box is empty, the match count usually < 30 MatchPoint
            if Match_Count > 24:
                #Print running time
                print(PrintingElement[index_element_arr]+" VND") #After run all dataset, print to console which money was detected
             
            else:
                print("BOX IS EMPTY")
            if os.path.exists("img_input.png"):
                  os.remove("img_input.png")
            else:
                  print("The file does not exists")   
cam.release()

cv2.destroyAllWindows()


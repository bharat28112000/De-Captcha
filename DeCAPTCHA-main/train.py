import os
import cv2
import numpy as np

path = '/content/drive/MyDrive/Saqeeb/CS771ML/train'
imageList = os.listdir(path)
imageList.remove("labels.txt")
with open("/content/drive/MyDrive/Saqeeb/CS771ML/train/labels.txt", 'r') as file:
  Labels = file.readlines()
Labels = [line.rstrip('\n') for line in Labels]

Xtrain = []
ytrain = []

Xtest = []
ytest = []

trainRatio = 1

count = len(imageList)*trainRatio
i = 0 
for imageFile in imageList:
  if(i<count):
    #  add to train list
    Xtrain.append(imageFile)
    labelIndex = int(imageFile.split(".")[0])
    ytrain.append(Labels[labelIndex])
  else:
    Xtest.append(imageFile)
    labelIndex = int(imageFile.split(".")[0])
    ytest.append(Labels[labelIndex])
  i += 1

labelTonum = {'ALPHA' : 0,
 'BETA' : 1,
 'CHI' : 2,
 'DELTA' : 3,
 'EPSILON': 4,
 'ETA' : 5,
 'GAMMA' : 6,
 'IOTA' : 7,
 'KAPPA' : 8,
 'LAMDA': 9,
 'MU' :10,
 'NU' : 11,
 'OMEGA' : 12,
 'OMICRON':13,
 'PHI' : 14,
 'PI' : 15,
 'PSI' : 16,
 'RHO' : 17,
 'SIGMA' : 18,
 'TAU' : 19,
 'THETA' : 20,
 'UPSILON' : 21,
 'XI' : 22,
 'ZETA': 23}

def myfun():
  for imageIndex in range(len(Xtrain)):
    img = cv2.imread(os.path.join(path, Xtrain[imageIndex]))
    y_dim, x_dim = img.shape[:2]
    corners = np.array([img[0,0] ,img[0,-1] ,img[-1,0] ,img[-1,-1] ])
    unique, counts = np.unique(corners, axis = 0,  return_counts = True)
    backgnd = unique[np.argmax(counts)]
    background = np.where((img[:,:,0]==backgnd[0]) & (img[:,:,1]==backgnd[1]) & (img[:,:,2]==backgnd[2]))
    img[background] = np.array([255,255,255], dtype = np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    gray = cv2.cvtColor(img_dilation, cv2.COLOR_BGR2GRAY)
    start = True
    countPrev = 0
    mylist = []
    for i in range(500):
      vpp = np.sum(gray[:,i] < 250)
      if vpp > 12:
        if start:
          # print(i)
          temp = i
          start = False
        countPrev += 1
      if vpp == 0:
        if start == False:
          # print(i)
          if countPrev > 30:
            sizediff = i-temp
            if (sizediff < 150):
              x_temp = temp - int((150 - sizediff)/2)
              x = x_temp if x_temp > 0 else 0
              x = x if x + 150 < 500 else 350
              y = 0
            else:
              x = i
              y = 0
            mylist.append((x , y, 150, 150))
            countPrev = 0
          start = True    
    listSize = len(mylist)
    if listSize != 3 :
      mylist = [(15 , 0 , 150, 150),(175, 0, 150, 150),(335, 0, 150, 150)]
    labelList = ytrain[imageIndex].split(",")
    numberLabel = []
    for labb in labelList:
      numberLabel.append(labelTonum[labb])
    for boxIndex in range(3):
      x, y = mylist[boxIndex][0], mylist[boxIndex][1]
      finalImage = gray[y:y + 150, x:x + 150]
      finalImage = cv2.resize(finalImage, (30, 30))
      finalImage = finalImage.flatten()
      dfX.append(finalImage)
      dfy.append(numberLabel[boxIndex])
  return "success"
dfX = []
dfy = []

print(myfun())

########## Logistic Regression ##########
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0,max_iter=4000)
classifier.fit(dfX,dfy)
svm_predictions = classifier.predict(dfX)
count = 0
for i in range(len(svm_predictions)):
    if svm_predictions[i] == dfy[i]:
        count += 1
print(count/(i+1)*100)
filename = 'my_model_small.sav'
pickle.dump(classifier, open(filename, 'wb'))
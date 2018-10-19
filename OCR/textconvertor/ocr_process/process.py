# #### Bounding box :                   
# 
# 
#       Name        : Bounding Box words split into character
#       Input       : Image
#       output      : Bounding boxes to all word (If output is not correct please change kernel according to image)
#       Description : bounding box is merely the coordinates of the rectangular border that fully encloses a digital image.
#                        (If output is not correct please change kernel and threshold according to image)
#       Author      : Rabbit and Tortoise solution private limited
#       Date        : -/09/2018
# 
from OCR.settings import MEDIA_ROOT
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import os
#import pytesseract
def get_contour_precedence(contour, cols):
    return contour[1] * cols + contour[0]

def get_text(words,count1):
    curr_dir=os.getcwd()
    if os.path.exists(curr_dir+"\\image"):
        pass
    else:
        os.makedirs(curr_dir+"\\image")
    if os.path.exists(curr_dir+"\\characters"):
        pass
    else:
        os.makedirs(curr_dir+"\\characters")    
    contr=curr_dir+"\\image"
    char=curr_dir+"\\characters"
    word=curr_dir+"\\words"
    imgTrainingNumbers = words
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray,(3,5), 0)                        
    imgThresh = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,4)
    imgThreshCopy = imgThresh.copy()
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours1=cv2.drawContours(imgGray,npaContours,-1,(0,0,255),0)
    count = 0
    label = count1
    list_sort = []
    for contour in npaContours:
        list_sort.append(cv2.boundingRect(contour))
        list_sort1 = sorted(list_sort,key = lambda x:x[0])
    for k in range(len(list_sort1)):
        x,y,w,h = list_sort1[k][0],list_sort1[k][1],list_sort1[k][2],list_sort1[k][3]
        contours2 = cv2.rectangle(imgTrainingNumbers,(x,y),(x+w,y+h),(0,0,255),0)
        contours2 = imgThresh[y:y+h,x:x+w]
        tmp=np.zeros((32,32,3))
        image = contours2
        w,h = image.shape[:2]  
        a = int((tmp.shape[0]- image.shape[0])/2) 
        b = int((tmp.shape[1]-image.shape[1])/2)
        if w and h <= 32:
            for i in range(w):   # 11
                for j in range(h): # 17
                    tmp[i+a][j+b] = image[i][j] 
        else:
            pass
        cv2.imwrite(char+"\\"+str(label)+'_'+str(count)+'.png',tmp)
        count+=1
def boundingBox(img, imFloodfillInv):
    curr_dir=os.getcwd()
    if os.path.exists(curr_dir+"\\words"):
        pass
    else:
        os.makedirs(curr_dir+"\\words")
    word=curr_dir+"\\words"    
    connectivity = 8
    labelnum, _, contours, _ = cv2.connectedComponentsWithStats(
        imFloodfillInv, connectivity)
    contours = sorted(contours, key=lambda x:get_contour_precedence(x, img.shape[0]))
    bb_img = img.copy()
    count = 0        
    for label in range(1,len(contours)):
        x,y,w,h,size = contours[label]
        bb_img = cv2.rectangle(bb_img, (x,y), (x+w,y+h), (0,0,255), 1)
        words = img[y:y+h,x:x+w]
        cv2.imwrite(word+"\\"+str(label)+".png",words)        
        all_character=get_text(words,count)
        count+=1
    # cv2.imwrite(r"C:/Users/balas/Desktop/OCR/textconvertor/media/output/BB_result.png",bb_img)
    cv2.imwrite(MEDIA_ROOT+"/output/BB_result.png",bb_img)

def TextProcessing(path):
    base_image =cv2.imread(path)
    gray_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype('uint8')
    Iedge = cv2.Canny(gray_image, 100, 200)
    kernel = np.ones((15,15), np.uint8)
    img_dilation = cv2.dilate(Iedge, kernel, iterations=1)
    th, im_th = cv2.threshold(img_dilation, 215, 255, cv2.THRESH_BINARY_INV);
    im_floodfill = im_th.copy()    
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    mask1 = np.zeros((h+10, w+10), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)        
    boundingBox(base_image, im_floodfill_inv)

# TextProcessing(name)


def startprocess(imagepath):
    path = MEDIA_ROOT+'\\'+str(imagepath)
    print('path is -------------',path)
    # print('---------imagepath-----------',path)
    TextProcessing(path)
    # output={'url':MEDIA_ROOT+'\\output\\BB_result.png'}
    output = '../output/BB_result.png'
    return output

# name = "C:\\Users\\balas\\Desktop\\OCR\\textconvertor\\media\\first_img.png"

# startprocess('/media/third_img_rhx3Gft.png')






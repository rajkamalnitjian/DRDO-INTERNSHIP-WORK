import argparse
from xml.etree.ElementPath import get_parent_map
import cv2
import imutils

import math
import csv
# import _LineStyle from matplotlib.lines 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas
# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

velocity=[]
def plotgraph(z):
   fig = plt.figure()
   ax = plt.axes()
   ax.plot(velocity, z)
   ax.set_xlabel('$velocity$', fontsize=20)
   ax.set_ylabel('$time$', fontsize=20)
   plt.show()
    
def velocity_calculation(d,height,fps,tv,foacl_length_camera_in_mm,vertical_dimesnion_of_image_in_mm,H):
    #    foacl_length_camera_in_mm=50
    #    vertical_dimesnion_of_image_in_mm=35
       tc=math.atan(vertical_dimesnion_of_image_in_mm/2*foacl_length_camera_in_mm)
    #    tv=60
       T=tv+(tc/2.0)

       D=H*math.tan(T)
       P=2*math.tan(tc/2)*math.sqrt(math.pow(H,2)+math.pow(D,2))
       K=P/int(height)
       t=1/int(fps)
      
       v=(3.6*K*d)/int(t)
       print(v)
       velocity.append(v)
       
       
points = pandas.read_csv('co-ordinate_for_use.csv')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = points['x_co-ordinate'].values
y = points['y_co-ordinate'].values
z = points['time'].values
print(type(z))
ax.legend()
ax.set_xlabel('$x_co-ordinate$', fontsize=20)
ax.set_ylabel('$y_co-ordinate$', fontsize=20)
ax.set_zlabel('$time$', fontsize=20)
ax.scatter(x, y, z, c='r', marker='o',s=1)
ax.plot(x, y, z, color='black')


plt.show()


stx=x[0]
sty=y[0]
velocity.append(0)
for i in range(1,len(x)):
    distance=math.sqrt(math.pow(x[i]-stx,2)+math.pow(y[i]-sty,2))
    stx=x[i]
    sty=y[i]
    velocity_calculation(distance,240,30,60,50,35,7.6)
plotgraph(z)  
    
   
   




















# # ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--image", required=True, help="image1.jpeg")
# # args = vars(ap.parse_args())

# # load image
# image = cv2.imread('image1.jpeg')

# # convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # blurred image
# blur = cv2.GaussianBlur(gray, (5,5), 0)

# # treshold 
# tresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY)[1]



# # find contours in the thresholded image
# cnts = cv2.findContours(tresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)

# for c in cnts:

#     # compute the center of the contour
#     M = cv2.moments(c)
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#     cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
    
#     # draw outlines
#     cv2.drawContours(image, [c], -1, (0,255,0), 2)

#     # draw text
#     cv2.putText(image, 'center', (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

#     # show image
#     cv2.imshow('img', image)
#     cv2.waitKey(0)# # Preprocessing is used to minimize bold shadow
# # # that can be detected as solid object which can decrease object
# # # detection accuracy

# # import cv2
# # import numpy as np
# # import   __future__
# # import os
# # import pygame
# # import time
# # import random
# # # from __future__ import print_function

# # import argparse
# # # img = cv2.imread('image1.jpeg', -1)
# # import cv2 
# # import numpy as np 

# # image =cv2.imread('image1.jpeg',-1)
# # # cv2.imshow('Original Image',image)
# # cv2.waitKey(0)
# # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# # # cv2.imshow('Gray Image',gray)
# # cv2.waitKey(0)
# # _,binary = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
# # cv2.imshow('Binary image',binary)
# # print(binary)
# # cv2.waitKey(0)
# # contours,hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# # drawing = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
# # CountersImg = cv2.drawContours(drawing,contours, -1, (255,255,0),3)
# # # cv2.imshow('Contours',CountersImg)
# # cv2.waitKey(0)

# # # def Shadow_Removal(img):
# # #     h = img.shape[0]
# # #     w = img.shape[1]
# # #     for y in range(0, h):
# # #          for x in range(0, w):
# # #               # threshold the pixel
# # #             # print(img[y,x][0])
# # #             for i in range(0,3):
# # #                  if img[y][x][i]>=0 and img[y][x][i]<=127:
# # #                     img[y][x][i]=0
# # #                  else:
# # #                     #  print('hello')
# # #                      img[y][x][i]=255
                     
                         
# # # Shadow_Removal(img)
# # # cv2.imshow('annoted_image', img)
# # # cv2.imwrite('Frame', frame)
# # # rgb_planes = cv2.split(img)

# # # result_planes = []
# # # result_norm_planes = []
# # # for plane in rgb_planes:
# # #     dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
# # #     bg_img = cv2.medianBlur(dilated_img, 21)
# # #     diff_img = 255 - cv2.absdiff(plane, bg_img)
# # #     norm_img = cv2.normalize(
# # #         diff_img, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
# # #     result_planes.append(diff_img)
# # #     result_norm_planes.append(norm_img)

# # # result = cv2.merge(result_planes)
# # # result_norm = cv2.merge(result_norm_planes)

# # # cv2.imwrite('shadows_out.png', result)
# # # cv2.imwrite('shadows_out_norm.png', result_norm)




# # # frame = cv2.imread('shadows_out_norm.png', -1)
# # # backSub = cv2.createBackgroundSubtractorMOG2()
# # # fgMask = backSub.apply(frame)


# # # # cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
# # # # cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
# # # #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

# # # # while():
# # # cv2.imshow('Frame', frame)
# # # cv2.imshow('FG Mask', fgMask)

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # # cv2.imwrite('FG Mask', fgMask)

# # #     # keyboard = cv2.waitKey(30)
# # #     # if keyboard == 'q' or keyboard == 27:
# # #     #    break

# from mpl_toolkits import mplot3d
# # matplotlib inline
# import numpy as np
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = plt.axes(projection='3d')

# listx=[]
# listy=[]
# listtime=[]
# for i  in range(1,3):
#    a,b = input().split("  ")
#    a=int(a)
#    b=int(b)
#    listx.append(a)
#    listy.append(b)
#    listtime.append(i)
# plt.scatter(listx, listy)
# plt.show()

# fig = plt.figure(figsize =(14, 9))
# ax = plt.axes(projection ='3d')

# ax.plot3D(listtime, listx,listtime, 'red')
# ax.legend()
# ax.set_xlabel('$time$', fontsize=20)
# ax.set_ylabel('$X$', fontsize=20)
# ax.set_zlabel('$Y$',fontsize=20)

# plt.show()
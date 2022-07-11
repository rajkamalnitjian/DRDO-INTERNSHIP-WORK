from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imutils
import keyboard
import csv
max_area = 1

import os
os.remove("co-ordinate.csv")

header = ['x_co-ordinate', 'y_co-ordinate', 'time']
import csv


with open('co-ordinate.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)



def myFunc(e):
    return len(e)


def print_pixels(img):
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            print(img[y][x])

    return img


def Shadow_Removal(img):
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            
            for i in range(0, 3):
                if img[y][x][i] >= 0 and img[y][x][i] <= 127:
                    img[y][x][i] = 0
                else:
                   
                    img[y][x][i] = 255
    return img


def Morphological_Operations(img):
   
    kernel = np.ones((5, 5), np.uint8)


    invert = cv.bitwise_not(img)


    dilation = cv.dilate(invert, kernel, iterations=1)


    plt.imshow(dilation, cmap='gray')
    return dilation


def median_filter(img, new_img):
    prop = img.shape

    for i in range(1, prop[0] - 1):
        for j in range(1, prop[1] - 1):

            win = []
            for x in range(i-1, i + 2):
                for y in range(j-1, j+2):
                    win.append(img[x][y])
            
            win.sort(key=myFunc)

            new_img[i][j] = win[4]

    cv.imwrite('3x3_median.jpg', new_img)
    return new_img


def Background_Subtraction(img):

    rgb_planes = cv.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv.medianBlur(dilated_img, 21)
        diff_img = 255 - cv.absdiff(plane, bg_img)
        norm_img = cv.normalize(
            diff_img, None, alpha=0, beta=200, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv.merge(result_planes)
    result_norm = cv.merge(result_norm_planes)

    cv.imwrite('shadows_out.png', result)
    cv.imwrite('shadows_out_norm.png', result_norm)

    return result_norm

def main(filepath):
    
        parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                    OpenCV. You can process both videos and images.')
        parser.add_argument('--input', type=str,
                            help='Path to a video or a sequence of image.', default=filepath)
        parser.add_argument('--algo', type=str,
                            help='Background subtraction method (KNN, MOG2).', default='MOG2')
        args = parser.parse_args()


        if args.algo == 'MOG2':
            backSub = cv.createBackgroundSubtractorMOG2()
        else:
            backSub = cv.createBackgroundSubtractorKNN()

        capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
        if not capture.isOpened():
            print('Unable to open: ' + args.input)
            exit(0)

        cnt = 0
        while True:
            ret, frame = capture.read()
            if frame is None:
                break
            if cv.waitKey(10) == 27:                    
                break
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            # if keyboard.is_pressed("q"):
            #     print("q pressed, ending loop")
            #     break
            

            gray = frame
            
            frame = Background_Subtraction(frame)
            
            
            fgMask = backSub.apply(frame)
            
            
            frame = median_filter(frame, frame)
            frame = Shadow_Removal(frame)
        
        
            cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
            cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            
            
            
            im = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            ret, im = cv.threshold(im, 100, 255, cv.THRESH_BINARY_INV)
            contours, hierarchy = cv.findContours(
                im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            im = cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
            # contours= contours[0] if imutils.is_cv2() else contours[1]
            cnts = sorted(contours, key=cv.contourArea, reverse=True)[:10]

            mainContour = None
            mainMoments = None
            contourCentreX = None
            contourCentreY = None

            maxArea = 0.0

            for c in cnts:
                area = cv.contourArea(c)
                if area > maxArea and area > 600 and area < 5000:  
                    M_2 = cv.moments(c)
                    cX = int(M_2['m10']/M_2['m00'])
                    cY = int(M_2['m01']/M_2['m00'])
                    maxArea = area
                    mainContour = c
                    M = cv.moments(c)
                    contourCentreX = int(M['m10']/M['m00'])
                    contourCentreY = int(M['m01']/M['m00'])
            if cnt%5==0:
              data=[]
              data.append(str(contourCentreX))
              data.append(str(contourCentreY))
              data.append(str(cnt))
              with open('co-ordinate.csv', 'a', encoding='UTF8', newline='') as f:
                  writer = csv.writer(f)
            
                  writer.writerow(data)
            
            print(str(contourCentreX)+'  '+str(contourCentreY))
            cnt+=1



from tkinter import *
from tkinter import filedialog
from tkinter import ttk


win = Tk()
win.geometry("700x350")


style=ttk.Style(win)
def open_win_diag():
 
   win.filename =  filedialog.askopenfilename(initialdir = "",title = "choose your file",filetypes = (("mp4 files","*.mp4"),("all files","*.*")))
   filepath=win.filename
   main(filepath)


label=Label(win, text= "Click the button to browse the file", font='Arial 15 bold')
label.pack(pady= 20)

button=ttk.Button(win, text="Open", command=open_win_diag)
button.pack(pady=5)

win.mainloop()

















































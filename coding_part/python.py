from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imutils
import keyboard

max_area = 1
listx = []
listy = []
listtime = []


def plot_function():

    # for i in range(1, len(listx)):
    #     a, b = input().split("  ")
    #     a = int(a)
    #     b = int(b)
    #     listx.append(a)
    #     listy.append(b)
    #     listtime.append(i)
    plt.scatter(listx, listy)
    plt.show()

    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection='3d')

    ax.plot3D(listtime, listx, listtime, 'red')
    ax.legend()
    ax.set_xlabel('$time$', fontsize=20)
    ax.set_ylabel('$X$', fontsize=20)
    ax.set_zlabel('$Y$', fontsize=20)

    plt.show()


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
            # threshold the pixel
            # print(img[y,x][0])
            for i in range(0, 3):
                if img[y][x][i] >= 0 and img[y][x][i] <= 127:
                    img[y][x][i] = 0
                else:
                    #  print('hello')
                    img[y][x][i] = 255
    return img


def Morphological_Operations(img):
    #  binr = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    # define the kernel
    kernel = np.ones((5, 5), np.uint8)

# invert the image
    invert = cv.bitwise_not(img)

# erode the image
    #  erosion = cv.erode(invert, kernel,
    #                 iterations=1)
    dilation = cv.dilate(invert, kernel, iterations=1)

# print the output
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
            # sort the values
            # win.sort()
            # print(win)
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


parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str,
                    help='Path to a video or a sequence of image.', default='Static_Dummy_Ejection (online-video-cutter.com).mp4')
parser.add_argument('--algo', type=str,
                    help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

# [create]
# create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
# [create]

# [capture]
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
# [capture]
cnt = 0
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    if cv.waitKey(10) == 27:                     # exit if Escape is hit
        break
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    if keyboard.is_pressed("q"):
        print("q pressed, ending loop")
        break
    # if cnt>=10:
    #     break
    # [apply]
    # update the background model

    gray = frame
    frame = Background_Subtraction(frame)
    fgMask = backSub.apply(frame)
    frame = median_filter(frame, frame)
    frame = Shadow_Removal(frame)
    # print(frame)
    # print_pixels(frame)
    # frame=Morphological_Operations(frame)

    # print_pixels(frame)
    # [apply]

    # [display_frame_number]
    # get the frame number and write it on the current frame
    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    # [display_frame_number]

    # [show]
    # #show the current frame and the fg masks
    # cv.imshow('Frame', frame)
    # print(frame)
    # contours, hierarchy = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    ######################
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

    # loops through all contours detected and narrows possible pupil by area size and location
    for c in cnts:
        area = cv.contourArea(c)
        if area > maxArea and area > 600 and area < 5000:  # ensure the correct contour is detected
            M_2 = cv.moments(c)
            cX = int(M_2['m10']/M_2['m00'])
            cY = int(M_2['m01']/M_2['m00'])
            # if cX >= topLeftCornerX and cY >= topLeftCornerY and cX <= bottomRightCornerX and cY <= bottomRightCornerY:
            maxArea = area
            mainContour = c
            M = cv.moments(c)
            contourCentreX = int(M['m10']/M['m00'])
            contourCentreY = int(M['m01']/M['m00'])

    print(str(contourCentreX)+'  '+str(contourCentreY))
    listx.append(contourCentreX)
    listy.append(contourCentreY)
    listtime.append(cnt)
    # cnt+=1

    # print(contours)
    # for cnt in contours:
    #     M = cv.moments(cnt)
    #     area=cv.contourArea(cnt)
    #     if M["m00"] != 0:
    #        cX = int(M["m10"] / M["m00"])
    #        cY = int(M["m01"] / M["m00"])
    #     else:
    # # set values as what you need in the situation
    #        cX, cY = 0, 0
    #     if area>1000:
    #         M = cv.moments(cnt)
    #         cx = float(M['m10']/M['m00'])
    #         cy = float(M['m01']/M['m00'])

    #         center=(cX,cY)
    #         print("Center coordinate: "+str(center))
    #     else:
    #        print('hello')
    # # set values as what you need in the situation
    #        cX, cY =0.0,0.0

    # draw the contour and center of the shape on the image
    # cv.drawContours(img_as_ubyte(im), [c], -1, (0, 255, 0), 2)
    # cv.circle(img_as_ubyte(im), (cX, cY), 7, (255, 255, 255), -1)
    # cv.putText(img_as_ubyte(im), "center", (cX - 20, cY - 20),
    # cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# viewer = ImageViewer(image)
# viewer.show()
    ######
    # cnt = contours[4]
    # im=cv.drawContours(frame, [cnt], 0, (0,255,0), 3)
    ####
    # cv.imshow('image', im)
    # print(contours.shape())
    ######################
    # drawing = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    # CountersImg = cv.drawContours(drawing,contours, -1, (255,255,0),3)
    # cv.imshow('Contours',CountersImg)
    # cv.waitKey(0)
    # cv.imshow('FG Mask', fgMask)
    # [show]
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # keyboard = cv.waitKey(30)
    # if keyboard == 'q' or keyboard == 27:
    #     break
plot_function()












from tkinter import * 
from tkinter import messagebox

import numpy as np
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
       K=P/height
       t=1/fps
      
       v=(3.6*K*d)/t
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
       
       
       
     
root = Tk() 
root.title("Age range validator")
root.geometry("500x350+0+0") 
# fps=0
# tv=0
# focal=0
# height=0
# H=0

heading = Label(root, text="Lets see trajectory and velocity graph!", font=("arial", 15, "bold"), fg="red").pack() #creates a heading of text
label_frame_speed = Label(root, text="Frame speed :", font=("arial", 10, "bold"), fg="green").place(x=10, y=50)
label_angle_of_camera = Label(root, text="Angle of Camera:", font=("arial", 10, "bold"), fg="green").place(x=10, y=100)
label_focal_length = Label(root, text="focal length of camera:", font=("arial", 10, "bold"), fg="green").place(x=10, y=150)
vertical_dimesnion_of_image=Label(root,text="vertical_dimesnion_of_img:",font=("arial",10,"bold"),fg="green").place(x=10,y=200)
dist_bw_cam_object=Label(root,text="dist bw cam and objects:",font=("arial",10,"bold"),fg="green").place(x=10,y=250)

vertical_of_frame=Label(root,text="vertical_of_frame",font=("arial",10,"bold"),fg="green").place(x=10,y=300)



fspeed = Entry(root, width=25, bg="white")
fspeed.place(x=340, y=50) 

camera_angle= Entry(root, width=25, bg="white")
camera_angle.place(x=340, y=100)

focal_length= Entry(root, width=25, bg="white")
focal_length.place(x=340, y=150)

vdim_cam=Entry(root,width=25,bg="white")
vdim_cam.place(x=340,y=200)

d_value=Entry(root,width=25,bg="white")
d_value.place(x=340,y=250)
height=Entry(root,width=25,bg="white")
height.place(x=340,y=300)
     





def check_age(): 
 
    print('hello')
    fps=float(fspeed.get())
    tv=float(camera_angle.get())
    focal=float(focal_length.get())
    Vertcam=float(vdim_cam.get())
    D=float(d_value.get())
    height1=float(height.get())
    
    
    stx=x[0]
    sty=y[0]
    velocity.append(0)
    for i in range(1,len(x)):
        distance=math.sqrt(math.pow(x[i]-stx,2)+math.pow(y[i]-sty,2))
        stx=x[i]
        sty=y[i]
        velocity_calculation(distance,height1,fps,tv,focal,Vertcam,D)
    plotgraph(z)  
    
 
# print(fps+tv+focal+V+H)
 
check = Button(root, text="Execute", width=10, height=2, bg="white", command= check_age).place(x=120, y=400) #creates the button and calls the function to be executed

root.mainloop() 

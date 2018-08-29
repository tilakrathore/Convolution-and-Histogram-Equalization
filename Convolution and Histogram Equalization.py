#################  Code for Q1  #################

import cv2                             # library declaration
import numpy as np
from matplotlib import pyplot as plot
img = cv2.imread('C:\lena_gray.jpg',0)

pad_img = cv2.copyMakeBorder(img, 1,1,1,1, cv2.BORDER_CONSTANT, value = 0)  #image padding
height, width=pad_img.shape

#######   2d_convolution   ########
       
convol_Gx_2d = np.zeros((height,width))            #array declaration for 2_d and 1_d                                                                             
convol_Gy_2d = np.zeros((height,width))
convol_G_2d = np.zeros((height,width))
convol_Gx_1d =  np.zeros((height,width))
convol_Gy_1d =  np.zeros((height,width))
convol_G_1d =  np.zeros((height,width))
convol_Gx_r = np.zeros((height,width))
convol_Gy_r = np.zeros((height,width))
Gx_c = np.array([[1],[2],[1]])                     # filters declaration
Gx_r = np.array([-1,0,1])
Gy_c = np.array([[-1],[0],[1]])
Gy_r = np.array([1,2,1])
Gx_2d = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Gy_2d = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
   
for i in range (1,height-1):                                      # for loop for 2d convolution
    for j in range (1,width-1):
        convol_Gx_2d[i][j] = np.sum(pad_img[i-1:i+2,j-1:j+2]*Gx_2d)
        convol_Gy_2d[i][j] = np.sum(pad_img[i-1:i+2,j-1:j+2]*Gy_2d)
    
convol_Gx_2d = convol_Gx_2d/convol_Gx_2d.max()                         # normalization of Gx,Gy,G
convol_Gy_2d = convol_Gy_2d/convol_Gy_2d.max()   
convol_G_2d = np.sqrt((convol_Gx_2d*convol_Gx_2d)+(convol_Gy_2d*convol_Gy_2d)) 
    
cv2.imshow('image 2D_Gx',convol_Gx_2d)
cv2.imshow('image 2D_Gy',convol_Gy_2d)
cv2.imshow('image 2D_G',convol_G_2d)
                     
########## 1d_convolution ###########

for i in range (1,height-1):                                 # row convolution on Gx,Gy
    for j in range (1,width-1):
        convol_Gx_r[i][j] = np.sum(pad_img[i, j-1:j+2]*Gx_r)
        convol_Gy_r[i][j] = np.sum(pad_img[i, j-1:j+2]*Gy_r)
        
for i in range (1,height-1):                                # coloumn convolution on Gx,Gy
    for j in range (1,width-1): 
                                        
        Gx1 = convol_Gx_r[i-1][j]   
        Gx2 = convol_Gx_r[i][j]    
        Gx3 = convol_Gx_r[i+1][j] 
        Gy1 = convol_Gy_r[i-1][j]   
        Gy2 = convol_Gy_r[i][j]    
        Gy3 = convol_Gy_r[i+1][j] 
        Vx = np.array([[Gx1],[Gx2],[Gx3]])
        convol_Gx_1d[i][j] = np.sum(Vx*Gx_c) 
        Vy = np.array([[Gy1],[Gy2],[Gy3]])
        convol_Gy_1d[i][j] = np.sum(Vy*Gy_c)
        
                  
convol_Gx_1d = convol_Gx_1d/convol_Gx_1d.max()                   # normalization
convol_Gy_1d = convol_Gy_1d/convol_Gy_1d.max()   
convol_G_1d = np.sqrt((convol_Gx_1d*convol_Gx_1d)+(convol_Gy_1d*convol_Gy_1d)) 
convol_G_1d = convol_G_1d/convol_G_1d.max()

cv2.imshow('image 1D_Gx',convol_Gx_1d)
cv2.imshow('image 1D_Gy',convol_Gy_1d)
cv2.imshow('image 1D_G1',convol_G_1d)

####################   Code for Q2   ################  

img = cv2.imread('C:\Q2.jpg',0)  
cv2.imshow("Original image", img)
height,width=img.shape

count_array = np.zeros(256,dtype=np.int)          #array declarations
count_array_new = np.zeros(256,dtype=np.int)
trans_func = np.zeros(256,dtype=np.int)
final_array = np.zeros(256,dtype=np.int)
image_array = np.zeros((height,width),np.uint8)

for i in range (0,height):                       
    for j in range (0,width):
        intensity = int(np.around(img[i][j]))
        count_array[intensity] = count_array[intensity]+1
       
intensity_value = np.arange(0,256)
a = plot.figure(1)
plot.title('original histogram image')            #original histogram plot
plot.xlabel("Intensity value")
plot.ylabel("pixels count")  
plot.bar(intensity_value,count_array)
a.show()

count_array_new[0] = count_array[0]
for i in range (1,256):
    count_array_new[i] = count_array_new[i-1] + count_array[i]
b = plot.figure(2)
plot.title('Cumulative histogram')              #Cumulative histogram plot  
plot.xlabel("Intensity value")
plot.ylabel("pixels count")  
plot.bar(intensity_value,count_array_new)
b.show()  

for i in range (0,256):
    trans_func[i] = np.around(((256-1)*count_array_new[i])/(height*width))

c = plot.figure(3)
plot.title('transfer function')
plot.xlabel("Intensity value")
plot.ylabel("transformation function")         # Transfer function plot
plot.plot(intensity_value,trans_func)
c.show()
    
for i in range (0,height):
    for j in range (0,width):
          intensity_old = int(np.around(img[i][j]))
          image_array[i][j] = trans_func[intensity_old]
          intensity_new = image_array[i][j]
          final_array[intensity_new] = final_array[intensity_new]+1
d = plot.figure(4)
plot.title('enhanced histogram image')         #enhanced histogram plot
plot.xlabel("Intensity value")
plot.ylabel("pixels count")
plot.bar(intensity_value,final_array)
d.show()  
cv2.imshow("enhanced image",image_array)

        
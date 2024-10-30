import cv2
import numpy as np

image=cv2.imread("C:/Users/PC/Desktop/SII M2/Projets/Vision/Vison_project/objet1PNG_SII_VISION/001.png",-1)
arraynd=np.asarray(image)
print(image)
print("\n------------------------------------------")
print(arraynd)
print("\n------------------------------------------")
print(image[:,:,:1])
print("\n------------------------------------------")
print(image[:,:,1:2])
print("\n------------------------------------------")
print(image[:,:,2:])
print("\n--------------------************************************----------------------")
print(image[0:1,:,:])
print("\n------------------------------------------")
print(image[1:2,:,:])
print("\n------------------------------------------")
print(image[2:,:,:])
print("\n------------------------------------------*****************--------------***********---------")
print(image[:,:,0])
print(image[:,:,1])
print(image[:,:,2])

h,w,c=image.shape
for y in range(h):
    for x in range(w):
        print(str(image.item(y,x,0))+","+str(image.item(y,x,1))+","+str(image.item(y,x,2)))
print(image[200,200])
res=image.astype("float32")
print(res)

print(res.dtype)


res[:,:,0]=res[:,:,0]/2.1503
res[:,:,1]=res[:,:,1]/1.5873 
res[:,:,2]=res[:,:,2]/1.3000 

for y in range(h):
    for x in range(w):
        print(str(res.item(y,x,0))+","+str(res.item(y,x,1))+","+str(res.item(y,x,2)))
print(res[200,200])

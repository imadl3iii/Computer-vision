import cv2
import numpy as np

Path_lightSources="C:/Users/PC/Desktop/SII M2/Projets/Vision/Vison_project/objet1PNG_SII_VISION/light_directions.txt"
Path_intensitie="C:/Users/PC/Desktop/SII M2/Projets/Vision/Vison_project/objet1PNG_SII_VISION/light_intensities.txt"
Path_mask="C:/Users/PC/Desktop/SII M2/Projets/Vision/Vison_project/objet1PNG_SII_VISION/mask.png"
path_filename="C:/Users/PC/Desktop/SII M2/Projets/Vision/Vison_project/objet1PNG_SII_VISION/filenames.txt"
path_images="C:/Users/PC/Desktop/SII M2/Projets/Vision/Vison_project/objet1PNG_SII_VISION"

#********************************************Etape 01******************************************************************
def load_lightSources (light_directionsPath):
    print("start_load_lightSources")
    # ouverture du fichier light_directions et chargement des positions dans une liste de string light_directions_list
    light_directions_list = open(light_directionsPath, "r", encoding="utf-8").read().splitlines()
    # Récupération des valeurs des positions (x,y,z)
    lightSourses=[]
    for case in light_directions_list:
        valeurs_string=case.split(" ")
        list=[float(valeurs_string[0]),float(valeurs_string[1]),float(valeurs_string[2])]
        lightSourses.append(list)
    #convertion a une matrice numpy
    lightSourses=np.array(lightSourses)
    print("end_load_lightSources\n\n")
    return lightSourses

def load_intensSources(light_intensitiesPath):
    print("start_load_intensSources")
    # ouverture du fichier light_intensities et chargement des valeurs des intensités dans une liste de string light_directions_list
    light_intensities_list = open(light_intensitiesPath, "r", encoding="utf-8").read().splitlines()
    # Récupération des valeurs des intensitées (R,G,B) dans intensSources
    intensSources=[]
    for case in light_intensities_list:
        valeurs_string=case.split(" ")
        list=[float(valeurs_string[0]),float(valeurs_string[1]),float(valeurs_string[2])]
        intensSources.append(list)
    #convertion a une matrice numpy
    intensSources=np.array(intensSources)
    print("end_load_intensSources\n\n")
    return intensSources

def load_objMask(pathImageMask):
    print("start_load_objMask")
    imgMask=cv2.imread(pathImageMask,cv2.IMREAD_GRAYSCALE)
    """h,w = img.shape
    imgMask = np.zeros(img.shape,np.uint8)
    for y in range(h):
      for x in range(w):
          if(img[y][x]==0):
              imgMask[y][x]=0
          else:
               imgMask[y][x]=1"""
    print("end_load_objMask\n\n")
    return imgMask

def loadImages(pathImagesnames,pathImages):
    print("start_loadImages")
     # opening filenames that contient all images file name
    imagesFile_names_list = open(pathImagesnames, "r", encoding="utf-8").read().splitlines()
    imagesListe=[]
    for name in imagesFile_names_list:
        img=cv2.imread(pathImages+"/"+name,-1)
        imagesListe.append(img)
    print("end_loadImages\n\n")
    return imagesListe


#************************************************Etape 02***********************************************************
def Transformation():
    #Data loading
    mask=load_objMask(Path_mask)
    list=loadImages(path_filename,path_images)
    intensities=load_intensSources(Path_intensitie)
    h,w = mask.shape
    E=[]
    #creation of matrix E that containt all N images, one image in each line h*w cols each cells containt gray level of the pixel
    for i in range(len(list)):
        #select the image i format uint16
        image=list[i]
        #select intensité of image i
        intenslist=intensities[i]
        #create image format float32 for convertion 
        img32 = np.zeros((h,w,3),np.float32)
        #create image for the gray level
        imgNVG = np.zeros(mask.shape,np.float32)
        imageToligne=[]
        for y in range(h):
           for x in range(w):
               if(mask[y][x]==255):
                 intenseR=intenslist[0]
                 intenseG=intenslist[1]
                 intenseB=intenslist[2]
                 img32.itemset(y,x,0,(np.float32(image.item(y,x,0))/intenseB))
                 img32.itemset(y,x,1,(np.float32(image.item(y,x,1))/intenseG))
                 img32.itemset(y,x,2,(np.float32(image.item(y,x,2))/intenseR))
                 nvg=0.3*img32.item(y,x,2)+0.59*img32.item(y,x,1)+0.11*img32.item(y,x,0)
                 imgNVG.itemset(y,x,nvg/65535)
                 imageToligne.append(nvg/65535)
               else:
                   imageToligne.append(0)
        #imageToligne=np.reshape(imgNVG,(1,-1))
        #print(imageToligne.shape)
        E.append(imageToligne)
    E=np.array(E,np.float32)
    #print(E)
    return E
#********************************************************************************************
def transformation_Optimized():
    print("start_transformation_Optimized")
    #Data loading
    mask=load_objMask(Path_mask)
    list=loadImages(path_filename,path_images)
    intensities=load_intensSources(Path_intensitie)
    h,w = mask.shape
    E=np.zeros((len(list),h*w),np.float32)
    #creation of matrix E that containt all N images, one image in each line h*w cols each cells containt gray level of the pixel
    for i in range(len(list)):
        #select the image i format uint16
        image=list[i]
        #select intensité of image i
        intenslist=intensities[i]
        intenseR=intenslist[0]
        intenseG=intenslist[1]
        intenseB=intenslist[2]
        #create image format float32 for convertion 
        image=image.astype("float32")
        #NORMALIZATION
        image[:,:,0]=image[:,:,0]/intenseB
        image[:,:,1]=image[:,:,1]/intenseG
        image[:,:,2]=image[:,:,2]/intenseR
        imgNVG = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        imgNVG = imgNVG / 65535
        imgNVG=np.reshape(imgNVG,(1,-1))
        E[i,:]=imgNVG
        #print("\n************************************************************************")
        #print(E[i,:])
        #print("\n************************************************************************")
    #print("=====================================================")
    #print(E)
    #print(E.shape)
    print("end_transformation_Optimized\n\n")
    return E

#******************************************Main()********************************************
print("Main.Start() stereo photometrie processing\n Loading....\n\n")
mask=load_objMask(Path_mask)
#creation de la matrice E qui contient tout les images chaque une dans une ligne
E=transformation_Optimized()
#chargement des valeur de la position de la source lumineuse 
S=load_lightSources(Path_lightSources)
S_1=np.linalg.pinv(S)
#Calcul needle Map 
N=np.dot(S_1,E)
#Normalisation du vecteur normal
normalized_N=np.zeros(N.shape,np.float32)
normalized_N[0,:] = N[0,:]/np.sqrt(N[0,:]**2+N[1,:]**2+N[2,:]**2)
normalized_N[1,:] = N[1,:]/np.sqrt(N[0,:]**2+N[1,:]**2+N[2,:]**2)
normalized_N[2,:] = N[2,:]/np.sqrt(N[0,:]**2+N[1,:]**2+N[2,:]**2)
"""for i in range(N.shape[1]):
    if(N[0][i]!=0 or N[1][i]!=0 or N[2][i]):
       normalized_N.itemset(0,i,(N.item(0,i)/np.sqrt(N.item(0,i)**2+N.item(1,i)**2+N.item(2,i)**2)))
       normalized_N.itemset(1,i,(N.item(1,i)/np.sqrt(N.item(0,i)**2+N.item(1,i)**2+N.item(2,i)**2)))
       normalized_N.itemset(2,i,(N.item(2,i)/np.sqrt(N.item(0,i)**2+N.item(1,i)**2+N.item(2,i)**2)))"""
#Affichage de l'image final 
h,w=mask.shape
Result=np.zeros((h,w,3),np.uint8)
cpt=0
for y in range(h):
    for x in range(w):
        if(mask[y][x]==255):
            #print("R:"+str(N.item(0,cpt))+" G:"+str(N.item(1,cpt))+" B:"+str(N.item(2,cpt)))
            #print("R:"+str(normalized_N.item(0,cpt))+" G:"+str(normalized_N.item(1,cpt))+" B:"+str(normalized_N.item(2,cpt)))
            Result.itemset((y,x,0),int(((normalized_N.item(2,cpt)+1)/2)*255))
            Result.itemset((y,x,1),int(((normalized_N.item(1,cpt)+1)/2)*255))
            Result.itemset((y,x,2),int(((normalized_N.item(0,cpt)+1)/2)*255))
        cpt +=1
print("Showing 3D image reconsturction result")
cv2.imshow("Stereo photometrie",Result)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("\n\nEnd.")
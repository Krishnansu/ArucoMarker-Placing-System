import cv2
import numpy as np
import cv2.aruco as aruco
import math


def findAruco(image):                                                  #Function to find ID & slope of given rotated Aruco Markers
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    key=getattr(aruco,f'DICT_5X5_250')
    arucoDict=aruco.Dictionary_get(key)
    arucoParam= aruco.DetectorParameters_create()

    (corners,ids,rejected)=cv2.aruco.detectMarkers(image,arucoDict,parameters=arucoParam)


    
    if(len(corners)>0):
        ids=ids.flatten()

        for(markerCorner,markerId) in zip(corners,ids):
            corners=markerCorner.reshape((4,2))
            (topLeft,topRight,bottomRight,bottomLeft)=corners

            topLeft=(int(topLeft[0]),int(topLeft[1]))
            topRight=(int(topRight[0]),int(topRight[1]))
            bottomLeft=(int(bottomLeft[0]),int(bottomLeft[1]))
            bottomRight=(int(bottomRight[0]),int(bottomRight[1]))


            dy=topRight[1]-bottomRight[1]
            dx=topRight[0]-bottomRight[0]
            slope=0
            if dx==0:
                slope=0
            else:
                slope=dy/dx      

    return(ids,slope)


def Arucocorners(image):               #Function to find corners of Aruco Marker (separate function created to suit the need in program)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    key=getattr(aruco,f'DICT_5X5_250')
    arucoDict=aruco.Dictionary_get(key)
    arucoParam= aruco.DetectorParameters_create()
    

    (corners,ids,rejected)=cv2.aruco.detectMarkers(image,arucoDict,parameters=arucoParam)


    
    if(len(corners)>0):
        ids=ids.flatten()

        for(markerCorner,markerId) in zip(corners,ids):
            corners=markerCorner.reshape((4,2))
            (topLeft,topRight,bottomRight,bottomLeft)=corners

            topLeft=(int(topLeft[0]),int(topLeft[1]))
            topRight=(int(topRight[0]),int(topRight[1]))
            bottomLeft=(int(bottomLeft[0]),int(bottomLeft[1]))
            bottomRight=(int(bottomRight[0]),int(bottomRight[1]))



            intcorn=(topLeft,topRight,bottomRight,bottomLeft)
        
    return(intcorn)
  


imgc = cv2.imread("D:\OpenCV\project\CVtask.jpg")        #Open given image with the various shapes 
img=cv2.resize(imgc,(0,0),fx=0.5,fy=0.5)                 #Resizing the image to make it optimum for viewing on screen


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray,30,150)
cont, hier = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)   #Finding Contours

img2 = img.copy()
centercorn={}
shapeAngle={}
c=1             #Counter to keep count of the square contours detected

for cnt in cont:                                               #To detect squares
    if cv2.contourArea(cnt) > 10000:
        peri = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,peri*0.0028,True)
        if len(approx) == 4:
            cv2.drawContours(img2,[approx],-1,(0,0,255),1)
                       
            c+=1
            
            if(c%2==0):                    #Two contours are detected for every square. This code skips one contour of each square 
                continue
        
            else:
                lefttop = (approx[0][0][0], approx[0][0][1])
                righttop = (approx[1][0][0], approx[1][0][1])
                rightbottom = (approx[2][0][0], approx[2][0][1]) 
                leftbottom = (approx[3][0][0], approx[3][0][1])
                corn=(lefttop,righttop,rightbottom,leftbottom)             #Stores corners of each square as a tuple
                centre=(int((lefttop[0]+rightbottom[0])/2),int((lefttop[1]+rightbottom[1])/2))       #Center of square
                centercorn[centre]=corn                        #Dictionary to store Center : Corners
                slp=(lefttop[1]-leftbottom[1])/(lefttop[0]-leftbottom[0])        #Slope of side each square to find angle of rotation 
                ang = math.degrees(math.atan(slp))
                shapeAngle[centre]=ang                             #Dictionary to store Center : Angle




squarecol={}
for i in centercorn.keys():
    col=tuple(img2[i[1],i[0]])                  #To detect colour of center pixel of square
    squarecol[col]=i                            #Dictionary to store Center: Colour

colours={"Black":(0, 0, 0),"Orange":(9, 127, 240),"Green":(79, 209, 146),"Pink-Peach":(210, 222, 228)}  #Dictionary storing colours and their BGR values

colAruco={"Green":1,"Orange":2,"Black":3,"Pink-Peach":4}       #Dictionary storing Aruco ID corresponding to colours where they have to be pasted

coorid={}
for c in colAruco:
    coorid[colAruco[c]]=squarecol[colours[c]]               #Dictionary storing Aruco IDs : Center of square where they are to be pasted


arucoCol={}                                      #Dictionary storing BGR values
for i in colours:
    for k in colAruco:
        if(i==k):
            arucoCol[colours[i]]=colAruco[k]
print(arucoCol)                      #Print BGR values: ArucoID



files=[r"D:\OpenCV\project\1.jpg",r"D:\OpenCV\project\2.jpg",r"D:\OpenCV\project\3.jpg",r"D:\OpenCV\project\4.jpg"]      #List storing File names of the Aruco Markers provided
arid={}
slope1=[]

c=0
for i in files:
    imga=cv2.imread(i)
    arid[i],x=findAruco(imga)                     #Storing ArucoID in a dictionary
    slope1.append(x)                              #Storing slope of the Aruco in a list

angle=np.arctan(slope1)                           #Finding angle from slope
for j in range(0,4):
    angle[j]=angle[j]*180/math.pi                 #Converting radian to degree

borderType = cv2.BORDER_CONSTANT
corner=[]
arucoid={}
for j in range(0,4):
    imgr=cv2.imread(files[j])
    h,w = imgr.shape[:-1]
    rot_point = w//2, h//2
    ang = angle[j]
    rot_mat = cv2.getRotationMatrix2D(rot_point,ang,1)
    rot = cv2.warpAffine(imgr,rot_mat,(w,h))                     #Rotating ArucoMarkers from tilted orientation to vertical orientation
    

    y=Arucocorners(rot)                                          #Obtaining corners of the rotated ArucoMarkers

    corner.append(y)                                             #Storing corner values as tuples in a list

    
    
    rotn=rot[min(int(corner[j][0][1]),int(corner[j][1][1])) : max(int(corner[j][2][1]),int(corner[j][3][1])) , min(int(corner[j][0][0]),int(corner[j][3][0])) : max(int(corner[j][1][0]),int(corner[j][2][0]))]
                                                                     #Trimming the ArucoMarkers using corner values
    arucoid[int(arid[files[j]])]=rotn                                #A dictionary to store Aruco ID : Trimmed Aruco Marker
    cv2.imshow(str(arid[files[j]]),rotn)                             #Display the finally reoriented and trimmed Aruco Markers
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    



imgm=cv2.imread("D:\OpenCV\project\CVtask.jpg")                       #Opening the given image
img3=cv2.resize(imgm,(0,0),fx=0.5,fy=0.5)                             #Resizing the image to make it optimum for viewing on screen
cv2.imshow("Original Image",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()


for i in range(1,5):                                                              
            
            blank = np.zeros([1300,1300,3],dtype =np.uint8)            #Creating a blank large image for preventing loss of image during rotation
            blank.fill(255)                                            #To make it white in colour
            blank[100:720,300:1177]=img3                               #Pasting the given image on the blank white image

            

            h,w = blank.shape[:-1]                          #Rotating the image to orient one of the sqaures in vertical orientation
            rot_point = w//2, h//2
            ang = shapeAngle[coorid[i]]
            rot_mat = cv2.getRotationMatrix2D(rot_point,ang,1)
            rot = cv2.warpAffine(blank,rot_mat,(w,h),borderValue=(255,255,255))

        
            gray = cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY)                       
            canny = cv2.Canny(gray,30,150)
            cont, hier = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)      #Finding contours

            rot2 = rot.copy()

        

            for cnt in cont:
                if cv2.contourArea(cnt) > 12000:
                    peri = cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt,peri*0.0028,True)
                    if len(approx) == 4:
                        cv2.drawContours(rot2,[approx],-1,(0,0,0),1)
                        x1, y1, w1, h1 = cv2.boundingRect(approx)

                        lefttop = (approx[0][0][0], approx[0][0][1])
                        righttop = (approx[1][0][0], approx[1][0][1])
                        rightbottom = (approx[2][0][0], approx[2][0][1]) 
                        leftbottom = (approx[3][0][0], approx[3][0][1])
                        corn=(lefttop,righttop,rightbottom,leftbottom)         #Storing corners of the detected square
                        
                        centre=(int((lefttop[0]+rightbottom[0])/2),int((lefttop[1]+rightbottom[1])/2))   #Center of sqaure
                        colour=tuple(rot2[centre[1],centre[0]])                                          #Colour of center of square

                        
                        
                        ratio1 = abs(float(leftbottom[0]-rightbottom[0])/w1)         #Finding ratio of length of side of square and bounding rectangle
                        if (ratio1==0):                                              #for vertical line
                            ratio1=abs(float(leftbottom[0]-lefttop[0])/w1)
                        
                        if (ratio1 < 1.1) and (ratio1> 0.90):                        #Ratio should be near 1 for a sqaure
                             if(arucoCol[colour]==i):                                #If the colour of square matches the corresponding Aruco ID

                                arucorot=cv2.resize(arucoid[i],(w1,h1))               #Resizing the Aruco to fit in the square
                                rot2[y1:y1+h1,x1:x1+w1] = arucorot                    #Pasting Aruco on the square
                                break


            h,w = rot2.shape[:-1]
            rot_point = w//2, h//2
            ang = shapeAngle[coorid[i]]
            rot_mat = cv2.getRotationMatrix2D(rot_point,-ang,1)
            rot3 = cv2.warpAffine(rot2,rot_mat,(w,h),borderValue=(255,255,255))       #Rotating the image back to original orientation
            img3=rot3[100:720,300:1177]                                               #Extracting the original image from the larger blank image
 
            

cv2.imshow("Final Image",img3)                            #Displaying Final image
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("final.jpg",img3)                             #Saving Final Image as "final.jpg"         
            
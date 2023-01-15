from array import array
from itertools import count
import imutils
from sqlite3 import converters
import threading
import jetson.inference
import jetson.utils
import cv2
from collections import deque
from imutils.video import VideoStream
from pypylon import pylon
import cv2
import time
from numba import jit, cuda
import cProfile
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt



@cuda.jit(nopython=True)
def process(frame, box_height=6, box_width=16):
    height, width, _ = frame.shape
    for i in range(0, height, box_height):
        for j in range(0, width, box_width):
            roi = frame[i:i + box_height, j:j + box_width]
            b_mean = np.mean(roi[:, :, 0])
            g_mean = np.mean(roi[:, :, 1])
            r_mean = np.mean(roi[:, :, 2])
            roi[:, :, 0] = b_mean
            roi[:, :, 1] = g_mean
            roi[:, :, 2] = r_mean
    gpu_frame = cv2.cuda_GpuMat()
    frame = gpu_frame.upload(frame)
    return frame


@cuda.jit(nopython=True)
def process(img, box_height=6, box_width=16):
    height, width, _ = img.shape
    for i in range(0, height, box_height):
        for j in range(0, width, box_width):
            roi = img[i:i + box_height, j:j + box_width]
            b_mean = np.mean(roi[:, :, 0])
            g_mean = np.mean(roi[:, :, 1])
            r_mean = np.mean(roi[:, :, 2])
            roi[:, :, 0] = b_mean
            roi[:, :, 1] = g_mean
            roi[:, :, 2] = r_mean
    
    gpu_frame = cv2.cuda_GpuMat()
    img = gpu_frame.upload(img)
    return img


cudacount = cv2.cuda.getCudaEnabledDeviceCount()
# print(cv2.getBuildInformation())

print(cudacount, "cudacount")
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
greenLower = (20, 100, 150)
greenUpper = (40, 200, 255)

greenLower2 = (20, 100, 50)
greenUpper2 = (40, 200, 255)
pts = deque(maxlen=50)


# # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
plotlistesi=[]
plotlistesi2=[]

totallist=[]
totallist2=[]

bouncepointlist=[]
bouncepointlist2=[]


trajcross=[]
trajcross2=[]

lowercenter=[]
trajlist=[]

lowercenter2=[]
trajlist2=[]
# # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

img5 = np.zeros ((480,640,1))
img6 = np.zeros ((480,640,1))
img7 = np.zeros ((480,640,1))



# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
camera.ExposureTime.SetValue(12000.0)
camera.BalanceWhiteAuto.SetValue("Continuous")
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# # # # # # # # # # # köşeler# # # # # #
center_coordinates1 = (240, 226)
center_coordinates2 = (156, 378)
center_coordinates3 = (450, 389)
center_coordinates4 = (428, 233)
#radius = 2
color = (0, 0, 255)
thickness = -1
# # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

ortalamax=[]
ortalamay=[]



# # # # # # # # # # PolyLines# # # # # ####### # # # # # # # # # # # # # # # # # # # # # #

pts = np.array([[316, 234], [285, 477], [637, 476], [567, 236]], np.int32)

pts = pts.reshape((-1, 1, 2))
isClosed = True
color2 = (255, 0, 0)
thickness2 = 2
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

pts2 = np.array([[322, 4], [348, 475], [638, 477], [637, 3]], np.int32)

pts2 = pts2.reshape((-1, 1, 2))
isClosed = True
color2 = (255, 0, 0)
thickness2 = 2


posListX = []
posListY = []


xList = [item for item in range(0, 640)]
yList = [item for item in range(0, 480)]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

posListX2 = []
posListY2 = []

xList2 = [item for item in range(0, 640)]
yList2 = [item for item in range(0, 480)]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

farklistesiic = []
farklistesidis = []

webcamic = []
webcamdis = []

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

fps_count = 0
fps_count2 = 0


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def degerlendirme(deger):
    global farklistesiic
    global farklistesidis

    x1 = int(deger[0])

    y1 = int(deger[1])

    deger=(x1,y1)

    # # # # # # # # # # PolyLines# # # # # ####### # # # # # # # # # # # # # # # # # # # # # #

    pts = np.array([[316, 234], [285, 477], [637, 476], [567, 236]], np.int32)

    pts = pts.reshape((-1, 1, 2))
    isClosed = True
    color2 = (255, 0, 0)
    thickness2 = 2

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    dist = cv2.pointPolygonTest(pts, (x1, y1), False)
    if dist == 1.0:
        farklistesiic.append(deger)
    elif dist == -1.0:
        farklistesidis.append(deger)

    print(deger, "degerrrrr")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def minima(plotlistesi):
    xxx = np.array(plotlistesi)
    yyy = xxx*np.random.randn(len(xxx))**2
    peaks = find_peaks(yyy, height = 1, threshold = 1, distance = 1)
    height = peaks[1]['peak_heights'] #list containing the height of the peaks
    
    roundlist=np.round(yyy).astype(int)
    output = np.round(height).astype(int)
    heightmax=int(np.round(max(height)))
    linevalue=int(np.where(heightmax==roundlist)[0])

    peak_pos = xxx[peaks[0]] 
    y2 = yyy*-1
    minima = find_peaks(y2, threshold = 1, distance = 1)
    min_pos = xxx[minima[0]]   #list containing the positions of the minima
    min_height = y2[minima[0]]   #list containing the height of the minima
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(xxx,yyy)
    ax.scatter(peak_pos, height, color = 'r', s = 10, marker = 'D', label = 'maxima')
    ax.scatter(min_pos, min_height*-1, color = 'gold', s = 10, marker = 'X', label = 'minima')
    ax.legend()
    ax.grid()
    # plt.show()
    plotname='myplot'+str(fps_count)+'.png'

    


    plt.savefig(plotname)
    plt.close()

    return linevalue

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



def webminima(plotlistesi):
    xxx = np.array(plotlistesi)
    yyy = xxx*np.random.randn(len(xxx))**2
    peaks = find_peaks(yyy, height = 1, threshold = 1, distance = 1)
    height = peaks[1]['peak_heights'] #list containing the height of the peaks
    
    roundlist=np.round(yyy).astype(int)
    heightmax=int(np.round(max(height)))
    linevalue=int(np.where(heightmax==roundlist)[0])

    peak_pos = xxx[peaks[0]] 
    y2 = yyy*-1
    minima = find_peaks(y2, threshold = 1, distance = 1)
    min_pos = xxx[minima[0]]   #list containing the positions of the minima
    min_height = y2[minima[0]]   #list containing the height of the minima
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(xxx,yyy)
    ax.scatter(peak_pos, height, color = 'r', s = 10, marker = 'D', label = 'maxima')
    ax.scatter(min_pos, min_height*-1, color = 'gold', s = 10, marker = 'X', label = 'minima')
    ax.legend()
    ax.grid()
    # plt.show()
    plotname='myplot'+str(fps_count)+'.png'


    # print(xxx)
    # print("---------------------------------------------")
    # print(roundlist)
    # print("---------------------------------------------")
    # print(output)
    # print("---------------------------------------------")
    # print(heightmax)
    # print("---------------------------------------------")
    # print(linevalue)

    plt.savefig(plotname)
    plt.close()

    return linevalue
    

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def webdegerlendirme(deger):
    global webcamic
    global webcamdis

    x1 = int(deger[0])

    y1 = int(deger[1])

    deger=(x1,y1)

    # # # # # # # # # # PolyLines# # # # # ####### # # # # # # # # # # # # # # # # # # # # # #

    pts = np.array([[322, 4], [348, 475], [638, 477], [637, 3]], np.int32)

    pts = pts.reshape((-1, 1, 2))
    isClosed = True
    color2 = (255, 0, 0)
    thickness2 = 2

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    dist = cv2.pointPolygonTest(pts, (x1, y1), False)
    if dist == 1.0:
        webcamic.append(deger)
    elif dist == -1.0:
        webcamdis.append(deger)

    print(deger, "deger2")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def basler():

    global img 
    global fps_count2
    crashesbas =[]
    img5 = np.zeros ((480,640,1))
    img6 = np.zeros ((480,640,1))
    img7 = np.zeros ((480,640,1))
    prev_frame_time2 = 1
    while True:

        fps_count2 += 1
        grabResult = camera.RetrieveResult(
          5000, pylon.TimeoutHandling_ThrowException)
        image = converter.Convert(grabResult)
        # image = process(image)
        img = image.GetArray()
        # gpu_img = cv2.cuda_GpuMat()
        # img = gpu_img.upload(img)
        img = imutils.resize(img, width=600)
        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        

        mask = cv2.inRange(hsv, greenLower2, greenUpper2)
        mask2 = cv2.erode(mask, None, iterations=2)
        mask3 = cv2.dilate(mask2, None, iterations=2)

        cnts = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        new_frame_time2 = time.time()   
        fps = 1/(new_frame_time2-prev_frame_time2)
        prev_frame_time2 = new_frame_time2 
        fps = int(fps)
        fps = str(fps)
        cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

            if radius > 10:
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)

                fps_count2 = 0

                posListX2.append(x)
                posListY2.append(y)
                plotlistesi.append(y)
                totallist.append([int(x),int(y)])


        if fps_count2 > 10 and len(posListY2) > 3:

            try:

                max_value = np.max(posListY2)
                result = np.where(posListY2 == max_value)
                result = int(result[0])

                if posListY2[result+1] <posListY2[result+2]:
                    result = result+1


                arrU_Y = posListY2[0:result+1]
                arrD_Y = posListY2[result+1:]

                arrU_X = posListX2[0:result+1]
                arrD_X = posListX2[result+1:]

                result3 = posListX2[result]

                toplok = ([result3, max_value])
                
                # if len(toplok) >= 3:
                degerlendirme(toplok)

                crashY =int((posListY2[result] + posListY2[result+1])/2)
                crashX =int((posListX2[result] + posListX2[result+1])/2)
                crash =[crashX, crashY]
                crashesbas.append(crash)


            except:
                pass

            try:
                A, B, C = np.polyfit(arrD_X, arrD_Y, 2)

                for x in xList2:

                    c = int(A * x ** 2 + B * x + C)
                    cv2.circle(img, (x, c), 3, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img5, (x,c), 3, (255,255,255), cv2.FILLED)	
            except:
                continue

            try:
                A, B, C = np.polyfit(arrU_X, arrU_Y, 2)

                for x in xList2:

                    y = int(A * x ** 2 + B * x + C)
                    cv2.circle(img, (x, y), 3, (100, 50, 75), cv2.FILLED)
                    cv2.circle(img6, (x,y), 3, (255,255,255), cv2.FILLED)

            except:
                continue

            numberofpoint=minima(plotlistesi)
            bouncepoint=totallist[numberofpoint]
            # print("bouncepoint "+ bouncepoint)
            bouncepointlist.append(bouncepoint)
            print(bouncepointlist)


            img7 = cv2.bitwise_and(img5,img6)
            img7=img7.astype(np.uint8)
            cnts = cv2.findContours(img7.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            gbx =0
           
            centers=[]
            if len(cnts)>0:
                for i in cnts:
                    
                    c1 = cv2.moments(cnts[gbx])
                    if c1["m00"] != 0:
                        cX = int(c1["m10"] / c1["m00"])
                        cY = int(c1["m01"] / c1["m00"])
                    else:

                        cX, cY = 0, 0
                    center1 = (cX, cY)
                    centers.append(center1)
                    gbx += 1
                location = (np.argmax(centers, axis=0))[1]
                lowercenter2.append(centers[location])
            if len(centers) ==0 and toplok and crash:
                norm = ((int((toplok[0] + crash[0])/2), int((toplok[1]+ crash[1])/2)))
                lowercenter2.append(norm)
             
            elif len(centers) ==0 and toplok:
                lowercenter2.append(toplok)
       
          
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            trajlist2.append(center)
            img5 = np.zeros ((480,640,1))
            img6 = np.zeros ((480,640,1))

            posListX2.clear()
            posListY2.clear()
            plotlistesi.clear()
            totallist.clear()



        for i in farklistesiic:
            cv2.circle(img, (i[0], i[1]), 5, (0, 255, 0), cv2.FILLED)
            
        for i in farklistesidis:
            cv2.circle(img, (i[0], i[1]), 5, (0, 0, 255), cv2.FILLED)

        for i in bouncepointlist:
            cv2.circle(img, (i[0], i[1]), 5, (255, 0, 0), cv2.FILLED)    
            
        
        cv2.imshow("Basler", img)

        cv2.waitKey(2)


def webcam():

    global frame
    global fps_count
    crashesweb =[]
    img2 = np.zeros ((480,640,1))
    img3 = np.zeros ((480,640,1))
    img4 = np.zeros ((480,640,1))
    prev_frame_time = 1
    while True:

        fps_count += 1

        _, frame = cap.read()

        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        

        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        new_frame_time = time.time()   
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time 
        fps = int(fps)
        fps = str(fps)
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)


        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                fps_count = 0

                posListX.append(x)
                posListY.append(y)
                plotlistesi2.append(y)
                totallist2.append([int(x),int(y)])

        if fps_count > 10 and len(posListY) > 3:
            
            try:

                max_value = np.max(posListY)
                result = np.where(posListY == max_value)
                result = int(result[0])

                if posListY[result+1] <posListY[result+2]:
                    result = result+1


                arrU_Y = posListY[0:result+1]
                arrD_Y = posListY[result+1:]

                arrU_X = posListX[0:result+1]
                arrD_X = posListX[result+1:]

                result3 = posListX[result]

                toplok = ([result3, max_value])
                
                webdegerlendirme(toplok)

                crashY =int((posListY[result] + posListY[result+1])/2)
                crashX =int((posListX[result] + posListX[result+1])/2)
                crash =[crashX, crashY]
                crashesweb.append(crash)

            except:
                pass
            

            try:
                A, B, C = np.polyfit(arrD_X, arrD_Y, 2)

                for x in xList:

                    c = int(A * x ** 2 + B * x + C)
                    cv2.circle(frame, (x, c), 5, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img2, (x,c), 5, (255,255,255), cv2.FILLED)	

            except:
                continue

            try:
                A, B, C = np.polyfit(arrU_X, arrU_Y, 2)

                for x in xList:

                    y = int(A * x ** 2 + B * x + C)
                    cv2.circle(frame, (x, y), 5, (100, 50, 75), cv2.FILLED)
                    cv2.circle(img3, (x,y), 5, (255,255,255), cv2.FILLED)	
                     
            except:
                continue


            numberofpoint=minima(plotlistesi2)
            bouncepoint=totallist2[numberofpoint]
            # print("bouncepoint "+ bouncepoint)
            bouncepointlist2.append(bouncepoint)
            print(bouncepointlist2)

            img4 = cv2.bitwise_and(img2,img3)
            img4=img4.astype(np.uint8)
            cnts = cv2.findContours(img4.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            gbx =0
            
            centers=[]
            if len(cnts)>0:
                for i in cnts:
                    
                    c1 = cv2.moments(cnts[gbx])
                    
                    if c1["m00"] != 0:
                        cX = int(c1["m10"] / c1["m00"])
                        cY = int(c1["m01"] / c1["m00"])
                    else:
                        cX, cY = 0, 0

                    center1 = (cX, cY)
                    centers.append(center1)
                    gbx += 1
                location = (np.argmax(centers, axis=0))[1]
                lowercenter.append(centers[location])
            if len(centers) ==0 and toplok and crash:
                norm = ((int((toplok[0] + crash[0])/2), int((toplok[1]+ crash[1])/2)))
                lowercenter.append(norm)
                
            elif len(centers) ==0 and toplok:
                lowercenter.append(toplok)

          
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            trajlist.append(center)
            img2 = np.zeros ((480,640,1))
            img3 = np.zeros ((480,640,1))


            posListX.clear()
            posListY.clear()
            plotlistesi2.clear()
            totallist2.clear()


   

        for i in webcamic:
            cv2.circle(frame, (i[0], i[1]), 5, (0, 255, 0), cv2.FILLED)
          
        for i in webcamdis:
            cv2.circle(frame, (i[0], i[1]), 5, (0, 0, 255), cv2.FILLED)
        
        for i in bouncepointlist2:
            cv2.circle(frame, (i[0], i[1]), 5, (255, 0, 0), cv2.FILLED)    
             
            
        cv2.imshow("ImageColor", frame)
        cv2.waitKey(2)


t1 = threading.Thread(target=basler)
t2 = threading.Thread(target=webcam)
t1.start()
t2.start()
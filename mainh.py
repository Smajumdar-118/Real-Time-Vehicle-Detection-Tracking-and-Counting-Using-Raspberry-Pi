import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import pyrebase
import datetime
import os
from dotenv import load_dotenv
from tracker import*
load_dotenv()



API_KEY = os.getenv('MY_API_KEY')
API_ID = os.getenv('MY_API_ID')
config = {
    "apiKey": API_KEY,
    "authDomain": "raspberrypidata-3a619.firebaseapp.com",
    "projectId": "raspberrypidata-3a619",
    "databaseURL" : "https://raspberrypidata-3a619-default-rtdb.firebaseio.com/",
    "storageBucket": "raspberrypidata-3a619.appspot.com",
    "messagingSenderId": "717931168998",
    "appId": API_ID ,
    "measurementId": "G-0CF8B985MH"
}

firebase = pyrebase.initialize_app(config)
database = firebase.database()



model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
# cap=cv2.VideoCapture('tf.mp4')
cap=cv2.VideoCapture(0)


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker=Tracker()
tracker1=Tracker()
tracker2=Tracker()
cy1=184
cy2=209
offset=8
upcar={}
downcar={}
countercarup=[]
countercardown=[]
downbus={}
counterbusdown=[]
upbus={}
counterbusup=[]
downtruck={}
uptruck={}
countertruckdown=[]
countertruckup=[]
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    
    list=[]
    list1=[]
    list2=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
           list.append([x1,y1,x2,y2])
          
        elif'bus' in c:
            list1.append([x1,y1,x2,y2])
          
        elif 'truck' in c:
             list2.append([x1,y1,x2,y2])
            

    bbox_idx=tracker.update(list)
    bbox1_idx=tracker1.update(list1)
    bbox2_idx=tracker2.update(list2)
    
    for bbox in bbox_idx:
        x3,y3,x4,y4,id1=bbox
        cx3=int(x3+x4)//2
        cy3=int(y3+y4)//2
        if cy1<(cy3+offset) and cy1 > (cy3 - offset):
            upcar[id1] = (cx3,cy3)
        if id1 in upcar :
            if cy2<(cy3+offset) and cy2 > (cy3 - offset):
                cv2.circle(frame,(cx3,cy3),4,(255,0,0),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                if countercarup.count(id1)==0 :
                    countercarup.append(id1)


        if cy2<(cy3+offset) and cy2 > (cy3 - offset):
            downcar[id1] = (cx3,cy3)
        if id1 in downcar :
            if cy1<(cy3+offset) and cy1 > (cy3 - offset):
                cv2.circle(frame,(cx3,cy3),4,(255,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                if countercardown.count(id1)==0 :
                    countercardown.append(id1)
              
  ################### Bus##########################################################                
    for bbox1 in bbox1_idx:
        x5,y5,x6,y6,id2 = bbox1
        cx4= int(x5+x6)//2
        cy4 = int(y5+y6)//2
        if cy1 < (cy4+offset) and cy1 > (cy4 - offset):
            upbus[id2] = (cx4,cy4)
        if id2 in upbus:
            if cy2 < (cy4+offset) and cy2 > (cy4 - offset):
                cv2.circle(frame,(cx4,cy4),4,(255,0,0),-1)
                cv2.rectangle(frame,(x5,y5),(x6,y6),(255,0,255),2)
                cvzone.putTextRect(frame,f'{id2}',(x5,y5),1,1)
                if counterbusup.count(id2) == 0 :
                    counterbusup.append(id2)

        if cy2 < (cy4+offset) and cy2 > (cy4 - offset):
            downbus[id2] = (cx4,cy4)
        if id2 in downbus:
            if cy1 < (cy4+offset) and cy1 > (cy4 - offset):
                cv2.circle(frame,(cx4,cy4),4,(255,0,255),-1)
                cv2.rectangle(frame,(x5,y5),(x6,y6),(255,0,0),2)
                cvzone.putTextRect(frame,f'{id2}',(x5,y5),1,1)
                if counterbusdown.count(id2) == 0 :
                    counterbusdown.append(id2)


###############################  TRUCK ########################################################

    for bbox2 in bbox2_idx:
        x7,y7,x8,y8,id3 = bbox2
        cx5= int(x7+x8)//2
        cy5 = int(y8+y8)//2
        if cy1 < (cy5+offset) and cy1 > (cy5 - offset):
            uptruck[id3] = (cx5,cy5)
        if id3 in uptruck:
            if cy2 < (cy5+offset) and cy2 > (cy5 - offset):
                cv2.circle(frame,(cx5,cy5),4,(255,0,255),-1)
                cv2.rectangle(frame,(x7,y7),(x8,y8),(255,0,0),2)
                cvzone.putTextRect(frame,f'{id3}',(x7,y7),1,1)
                if countertruckup.count(id3) == 0 :
                    countertruckup.append(id3)

        if cy2 < (cy5+offset) and cy2 > (cy5 - offset):
            downtruck[id3] = (cx5,cy5)
        if id3 in downtruck:
            if cy1 < (cy5+offset) and cy1 > (cy5 - offset):
                cv2.circle(frame,(cx5,cy5),4,(255,0,255),-1)
                cv2.rectangle(frame,(x7,y7),(x8,y8),(255,0,0),2)
                cvzone.putTextRect(frame,f'{id3}',(x7,y7),1,1)
                if countertruckdown.count(id3) == 0 :
                    countertruckdown.append(id3)



    cv2.line(frame,(1,cy1),(1018,cy1),(0,255,0),2)
    cv2.line(frame,(3,cy2),(1016,cy2),(0,0,255),2)
    cup=len(countercarup)
    cdown=len(countercardown)
    cbuusup = len(counterbusup)
    cbuusdown = len(counterbusdown)
    ctruckdown = len(countertruckdown)
    ctruckup = len(countertruckup)
    cvzone.putTextRect(frame,f'UpCar:-{cup}',(14,30),2,2)
    cvzone.putTextRect(frame,f'DownCar:-{cdown}',(14,82),2,2)
    cvzone.putTextRect(frame,f'UpTruck:-{ctruckup}',(14,132),2,2)
    cvzone.putTextRect(frame,f'UpBus:-{cbuusup}',(833,35),2,2)
    cvzone.putTextRect(frame,f'DownBus:-{cbuusdown}',(792,85),2,2)
    cvzone.putTextRect(frame,f'DownTruck:-{ctruckdown}',(756,135),2,2)

    ct = datetime.datetime.now()
    ct_string = ct.strftime("%Y-%m-%d %H:%M:%S")


    data = {
        "UpCar": cup,
        "DownCar": cdown,
        "UpBus" : cbuusup,
        "DownBus" : cbuusdown,
        "UpTruck" : ctruckup,
        "DownTruck" : ctruckdown,
        "Time" : ct_string
    }
    database.child("Camera").child("Data").set(data)





   
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

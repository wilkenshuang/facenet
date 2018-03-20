# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:55:47 2018

@author: gd
"""

from xmlrpc.client import ServerProxy
import xmlrpc.client
import socket
import time
import json
import base64
import cv2
import numpy as np
from PIL import Image

#画出人脸框
def drawBoxes(im, boxes,frame):
    faces=1
    for rectangle in boxes:
        cropped=im[int(rectangle[1]):int(rectangle[3]),int(rectangle[0]):int(rectangle[2])]
        #faces.append(cropped)
        cv2.imwrite('video/'+str(frame)+'_'+str(faces)+'.jpg',cropped)
        faces+=1
    #return im

def drawFaces(im,boxes):
    for rectangle in boxes:
        #cv2.putText(im,str(rectangle[4]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        cv2.rectangle(im,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(0,255,0),1)
        #for i in range(5,15,2):
        	#cv2.circle(im,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
    return im

#服务器地址及接口
ip="http://10.0.0.247:8080"
port=8080
server = ServerProxy(ip)

#利用摄像头捕捉图片
cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
#cap.set(cv2.CAP_PROP_FPS,120)
size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) 
fps=cv2.CAP_PROP_FPS
#fps=cap.get(cv2.CAP_PROP_FPS)
print(fps)
print(size)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')#DIVX, XVID, MJPG, X264, WMV1, WMV2
out = cv2.VideoWriter('D:/opencv-video/face.avi',fourcc,5,(size[1],size[0]))
f=1
while(True):
    #一帧一帧抓取图像
    ret,frame=cap.read()
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    #encode_param=[int(cv2.IMWRITE_JPEG_OPTIMIZE),1]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = np.array(imgencode)
    stringData = data.tostring()
    
    data=base64.b64encode(stringData)
    data=data.decode()
    #sock.send( str(len(stringData)).ljust(16))
    #sock.send( stringData )
    data_r=server.recognize(data)
    points=np.array(json.loads(data_r))
    #print(points)
    #res=drawFaces(frame, points)
    #res=drawBoxes(frame, points)
    #显示图片
    #data = np.fromstring(receive, dtype='uint8')
    #decimg=cv2.imdecode(data,1) 
    
    #out.write(frame)
    
    timeF = 10
    if f%timeF==0:
        drawBoxes(frame, points,f)
        #cv2.imwrite('video/'+str(f)+'.jpg',frame)
    f+=1
    print(f)
    cv2.waitKey(1)
    if f>100:
        break
    '''
    ch=cv2.waitKey(1)&0xff
    if ch==ord('q'):
        out.write(frame)
        break
    if ch==ord('c'):
        cv2.imwrite('./capture.jpg',frame)
        break
    '''
#drawBoxes(img,data_r)
#decimg=cv2.imdecode(data,1) 
#cv2.imwrite('./frame.jpg',decimg)
        
cap.release()
out.release()
cv2.destroyAllWindows()

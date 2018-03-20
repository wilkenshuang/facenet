# -*- coding: UTF-8 -*-
import sys
import tensorflow as tf
import numpy as np
import cv2
import heapq
import os
import profile
import base64
import time
import io
import json
import base64
from PIL import Image
import SimpleXMLRPCServer
import demo
import SocketServer
from SocketServer import ThreadingMixIn
from SimpleXMLRPCServer import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import caffe

class TXMLRPCServer(ThreadingMixIn,SimpleXMLRPCServer):
    pass

class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths=('/RPC2')

#构造函数
'''
caffe_model_path='/app/huang/face_recognition/mtcnn/model'
caffe.set_device(0)
caffe.set_mode_cpu()
PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)
'''
#class ImgRecog:
def recognize(string):
    string=string.encode()
    receive=base64.b64decode(string)
    data = np.fromstring(receive, dtype='uint8')
    decimg=cv2.imdecode(data,1)
    boundingboxes=demo.mtcnn(decimg)
    '''
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),50]
    result, imgencode = cv2.imencode('.jpg', img, encode_param)
    data = np.array(imgencode)
    #boundingboxes = np.array(boundingboxes)
    print(boundingboxes)
    stringData = boundingboxes.tostring()
    data=base64.b64encode(stringData)
    data=data.decode()
    type(data)
    '''
    boundingboxes=boundingboxes.tolist()
    output=[]

    res=json.dumps(boundingboxes)

    return res


#创建服务器，注册方法
port=8080
#server=TXMLRPCServer(("10.0.0.247",port),SimpleXMLRPCRequestHandler)
server=SimpleXMLRPCServer(("10.0.0.247",port),logRequests=True,allow_none=True)

server.register_function(recognize)

print("Listening on port %d" %port)
server.serve_forever()

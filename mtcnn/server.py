# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 22:32:25 2018

@author: wilke
"""

import socket
import cv2
import numpy

address = ('10.0.0.247', 8080)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(address)
s.listen(True)

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

conn, addr = s.accept()
while 1:
    length = recvall(conn,16)
    stringData = recvall(conn, int(length))
    data = numpy.fromstring(stringData, dtype='uint8')
    decimg=cv2.imdecode(data,1)
    cv2.imshow('SERVER',decimg)
    if cv2.waitKey(10) == 27:
        break

s.close()
cv2.destroyAllWindows()

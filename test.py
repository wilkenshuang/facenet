# -*- coding: UTF-8 -*-
"""
Created on Fri Mar 16 14:07:25 2018
Based on code from https://github.com/shanren7/real_time_face_recognition
@author: wilkenshuang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mtcnn.real_time_detect as rtd
import tensorflow as tf
import numpy as np
import argparse
import src.facenet as facenet
import src.face as face
import os
import math
import pickle
from sklearn.svm import SVC
import sys
import cv2
import time
import io
from PIL import Image
from scipy import misc
from src.align import detect_face
import json

def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)
    '''
    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)
    '''

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str)
    parser.add_argument('output_path', type=str)
    '''
    parser.add_argument('classifier_filename', default="./models/my_classifier.pkl"
        help='Classifier model file name as a pickle (.pkl) file. ' +
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--model', type=str, default="./models/"
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    '''
    return parser.parse_args(argv)

def main(args):
    image=cv2.imread(args.img_path)
    #facenet_model_checkpoint=args.model
    #classifier_model=args.classifier_filename
    face_recognition = face.Recognition()
    faces = face_recognition.identify(image)
    for i in faces:
        print(i.name)

    add_overlays(image, faces)
    cv2.imwrite(args.output_path,image)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

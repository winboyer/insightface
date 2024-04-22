import cv2
import sys
import time
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace

thresh = 0.8
scales = [1024, 1980]

count = 1

gpuid = 0
# detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
detector = RetinaFace('/root/jinyfeng/models/retinaface-model/R50', 0, gpuid, 'net3')

# img = cv2.imread('t1.jpg')
# img = cv2.imread('/root/jinyfeng/datas/20230801_faces_v2_2619id/ffcf5df2-d360-4a4b-bc18-ac6f9730547d@_470406342369098405.jpg')
# img = cv2.imread('/root/jinyfeng/datas/Solvay_conference_1927.png')
img = cv2.imread('/root/jinyfeng/datas/sensoro/sensoro_person_info/hly_511023.jpg')
print(img.shape)
im_shape = img.shape
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
#im_scale = 1.0
#if im_size_min>target_size or im_size_max>max_size:
im_scale = float(target_size) / float(im_size_min)
# prevent bigger axis from being more than max_size:
if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)

print('im_scale', im_scale)

scales = [im_scale]
flip = False

for c in range(count):
    faces, landmarks = detector.detect(img,
                                       thresh,
                                       scales=scales,
                                       do_flip=flip)
    print(c, faces.shape, landmarks.shape)

t1=time.time()
faces, landmarks = detector.detect(img,
                                       thresh,
                                       scales=scales,
                                       do_flip=flip)
t2 = time.time()
print('detector.detect time: %s ms' % ((t2-t1)*1000))
    
if faces is not None:
    print('find', faces.shape[0], 'faces')
    for i in range(faces.shape[0]):
        #print('score', faces[i][4])
        box = faces[i].astype(np.int)
        #color = (255,0,0)
        color = (0, 0, 255)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        crop_face = img[box[1]:box[3], box[0]:box[2]]
        if landmarks is not None:
            landmark5 = landmarks[i].astype(np.int)
            #print(landmark.shape)
            for l in range(landmark5.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color,
                           2)

    filename = './detector_test.jpg'
    print('writing', filename)
    cv2.imwrite(filename, img)
    cv2.imwrite('./detect_face.jpg', crop_face)

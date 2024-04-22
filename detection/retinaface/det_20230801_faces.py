import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace

thresh = 0.8
# thresh = 0.6

count = 1
gpuid = 1
# detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
detector = RetinaFace('/root/jinyfeng/models/R50', 0, gpuid, 'net3')


# img_folder = '/root/jinyfeng/datas/20230801_faces_v2_2619id'
# save_folder = '/root/jinyfeng/datas/20230801_v2_crop_faces'
# facedet_result = '/root/jinyfeng/datas/20230801_faces_v2_2619id_bbox_ldmark.txt'

img_folder = '/root/jinyfeng/datas/sensoro/20230823_faces'
save_folder = '/root/jinyfeng/datas/sensoro/20230823_faces_cropface4'
facedet_result = '/root/jinyfeng/datas/sensoro/20230823_faces_bbox_ldmark4.txt'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

fopen = open(facedet_result, 'w')

filelist = os.listdir(img_folder)

# print(len(filelist))
file_num=0
image_padded_num=0
det_face_file_num=0
for filename in filelist:
#     print(filename)
    ext_flag = filename.endswith(".jpg")
    if not ext_flag:
        continue
    file_num+=1
    img_path = os.path.join(img_folder, filename)
#     print(img_path)
    img = cv2.imread(img_path)
    print('img.shape=======', img.shape)
    im_shape = img.shape
    img_h = im_shape[0]
    img_w = im_shape[1]
    print('img.height, img.width=======', im_shape[0], im_shape[1])
#     scales = [1024, 1980]
#     scales = [1024, 1280]
    scales = [1024, 1024]
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
    

    if faces is not None:
        print('find', faces.shape[0], 'faces')
        det_face_file_num+=1
        for i in range(faces.shape[0]):
            #print('score', faces[i][4])
            box = faces[i].astype(np.int_)
#             box = faces[i].astype(np.int_)[:4]
            score = faces[i][4]
            print('box, score======',box, score)
#             cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            face_w=box[2]-box[0]
            face_h=box[3]-box[1]
            face_wh = abs(face_h-face_w)
            pad_x0,pad_y0,pad_x1,pad_y1=0,0,0,0
            if face_w>face_h:
                box[1]=box[1]-face_wh/2
                box[3]=box[3]+face_wh-face_wh/2
                if box[1]<0:
                    pad_y0=0-box[1]
                    box[1]=0
                if box[3]>img_h:
                    pad_y1=box[3]-img_h
            elif face_w<face_h:
                box[0]=box[0]-face_wh/2
                box[2]=box[2]+face_wh-face_wh/2
                if box[0]<0:
                    pad_x0=0-box[0]
                    box[0]=0
                if box[2]>img_w:
                    pad_x1=box[2]-img_w
            if pad_x0>0 or pad_x1>0 or pad_y0>0 or pad_y1>0:
                print('image padded')
                image_padded_num+=1
                img = cv2.copyMakeBorder(img, pad_y0, pad_y1, pad_x0, pad_x1,cv2.BORDER_CONSTANT,value=(255,255,255))

            crop_face = img[box[1]:box[3], box[0]:box[2]]
            linestr = filename+','+str(box[0])+','+str(box[1])+','+str(box[2])+','+str(box[3])+','+str(score)+','
            if landmarks is not None:
                landmark5 = landmarks[i].astype(np.int_)
                #print(landmark.shape)
                for l in range(landmark5.shape[0]):
                    if l<4:
                        linestr += str(landmark5[l][0])+','+str(landmark5[l][1])+','
                    else:
                        linestr += str(landmark5[l][0])+','+str(landmark5[l][1])+'\n'
            fopen.write(linestr)
            
            savepath = os.path.join(save_folder, filename)
            print('writing', savepath)
            cv2.imwrite(savepath, crop_face)
            
            break #只取最大/置信度最高的人脸


print('file_num=========', file_num)
print('image_padded_num========', image_padded_num)
print('det_face_file_num========', det_face_file_num)
fopen.close()        
        
        
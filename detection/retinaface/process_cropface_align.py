import cv2
import sys
import numpy as np
import datetime
import os
import glob
import argparse
from skimage import transform

def face_align_landmark(img, landmark, image_size=(112, 112), method="similar"):
    tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()
    src = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32
    )
    tform.estimate(landmark, src)
    # ndimage = transform.warp(img, tform.inverse, output_shape=image_size)
    # ndimage = (ndimage * 255).astype(np.uint8)
    M = tform.params[0:2, :]
    ndimage = cv2.warpAffine(img, M, image_size, borderValue=0.0)
#     if len(ndimage.shape) == 2:
#         ndimage = np.stack([ndimage, ndimage, ndimage], -1)
#     else:
#         ndimage = cv2.cvtColor(ndimage, cv2.COLOR_BGR2RGB)
    return ndimage


# img_folder = '/root/jinyfeng/datas/20230801_faces_v2_2619id'
# save_folder = '/root/jinyfeng/datas/20230801_v2_align_faces'
# facedetinfo_path = '/root/jinyfeng/datas/20230801_faces_v2_2619id_bbox_ldmark.txt'
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='face align crop')
    parser.add_argument('--img-folder', type=str, default='/data2/ossdata/mz/dy_down_faceimgs')
    parser.add_argument('--facedetinfo-path', type=str, default='/data2/ossdata/mz/dy_down_yolov7-faceinfo.txt')
    parser.add_argument('--save-folder', type=str, default='/data2/ossdata/mz/dy_wb_xhs_alignface')
    args = parser.parse_args()
    
    print('saving align crop face to {}'.format(args.save_folder))
#     savefolder = os.path.dirname(args.save_folder)
    savefolder = args.save_folder
    if savefolder != '' and not os.path.exists(savefolder):
        os.makedirs(savefolder)
        
    facedetinfo = open(args.facedetinfo_path, 'r')
    infolists = facedetinfo.readlines()
    facedetinfo.close()
    print('len(infolists)==========', len(infolists))
    idx=0
    image_padded_num=0
    for info_line in infolists:
#         print(info_line)
        faceinfo = info_line.split('\n')[0]
        filename = faceinfo.split(',')[0]
        ext_flag = filename.endswith(".jpg")
        if not ext_flag:
            continue
        idx+=1
        if idx%1000 == 0:
            print('processed {} images'.format(idx))
            
        img_path = os.path.join(args.img_folder, filename)
        img = cv2.imread(img_path)
#         print('file_num, filename, img.shape============', idx, filename, img.shape)
        im_shape = img.shape
        img_h = im_shape[0]
        img_w = im_shape[1]
        bbox = faceinfo.split(',')[1:6]
        ldmarks = faceinfo.split(',')[6:]
#         print('bbox, ldmarks===========',bbox, ldmarks)
        ldmarks = np.array(ldmarks).astype(int)
        ldmarks = ldmarks.reshape((5, 2))
        
        score = bbox[4]
        bbox = np.array(bbox)[:4]
        bbox = bbox.astype(np.int_)
#         print('bbox, ldmarks, score======',bbox, ldmarks, score)
        face_w=bbox[2]-bbox[0]
        face_h=bbox[3]-bbox[1]
        face_wh = abs(face_h-face_w)
        pad_x0,pad_y0,pad_x1,pad_y1=0,0,0,0
        if face_w>face_h:
            bbox[1]=bbox[1]-face_wh/2
            bbox[3]=bbox[3]+face_wh-face_wh/2
            if bbox[1]<0:
                pad_y0=0-bbox[1]
                bbox[1]=0
            if bbox[3]>img_h:
                pad_y1=bbox[3]-img_h
        elif face_w<face_h:
            bbox[0]=bbox[0]-face_wh/2
            bbox[2]=bbox[2]+face_wh-face_wh/2
            if bbox[0]<0:
                pad_x0=0-bbox[0]
                bbox[0]=0
            if bbox[2]>img_w:
                pad_x1=bbox[2]-img_w
        if pad_x0>0 or pad_x1>0 or pad_y0>0 or pad_y1>0:
            image_padded_num+=1
#             print('image padded')
#             print('bbox, ldmarks, score======',bbox, ldmarks, score)
            ldmarks[:,0] += pad_x0
            ldmarks[:,1] += pad_y0
#             print('pad_x0, pad_y0, ldmarks======', pad_x0, pad_y0, ldmarks)
            img = cv2.copyMakeBorder(img, pad_y0, pad_y1, pad_x0, pad_x1, cv2.BORDER_CONSTANT, value=(255,255,255))

        crop_face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        ldmarks[:,0] -= bbox[0]
        ldmarks[:,1] -= bbox[1]
        
        align_face = face_align_landmark(crop_face, ldmarks)
        savepath = os.path.join(savefolder, filename)
#         print('savepath===========', savepath)
        cv2.imwrite(savepath, align_face)
    
    print('file_num=========', idx)
    print('image_padded_num========', image_padded_num)
    
        
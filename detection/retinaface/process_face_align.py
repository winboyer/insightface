import cv2
import sys
import numpy as np
import datetime
import os
import glob
import argparse
from skimage import transform


def align(self, image, landmark):
    tform = trans.SimilarityTransform()
    avg_landmarks = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32
    )
    tform.estimate(landmark, avg_landmarks)
    matrix_similarity = tform.params[0:2, :]
    image_align = cv2.warpAffine(image, matrix_similarity, (self.image_size[1], self.image_size[0]))
    
    return image_align


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
#     parser.add_argument('--img-folder', type=str, default='/root/jinyfeng/datas/20230801_faces_v2_2619id')
#     parser.add_argument('--facedetinfo-path', type=str, default='/root/jinyfeng/datas/20230801_faces_v2_2619id_bbox_ldmark.txt')
# #     parser.add_argument('--save-folder', type=str, default='/root/jinyfeng/datas/20230801_v2_align_faces')
#     parser.add_argument('--save-folder', type=str, default='/root/jinyfeng/datas/20230801_v2_align_faces_v2')
    
#     parser.add_argument('--img-folder', type=str, default='/root/jinyfeng/datas/sensoro/sensoro_person_info')
    
#     parser.add_argument('--facedetinfo-path', type=str, default='/root/jinyfeng/datas/sensoro/sensoro_person_info_bbox_ldmark_retinaface.txt')
#     parser.add_argument('--save-folder', type=str, default='/root/jinyfeng/datas/sensoro/sensoro_person_info_alignface_retinaface')
    
#     parser.add_argument('--facedetinfo-path', type=str, default='/root/jinyfeng/datas/sensoro/sensoro_person_info_bbox_ldmark_yolov7-face.txt')
#     parser.add_argument('--save-folder', type=str, default='/root/jinyfeng/datas/sensoro/sensoro_person_info_alignface_yolov7face')

    # parser.add_argument('--img-folder', type=str, default='/root/jinyfeng/datas/sensoro/sensoro_quality_face')
    # parser.add_argument('--facedetinfo-path', type=str, default='/root/jinyfeng/datas/sensoro/sensoro_quality_face_bbox_ldmark_yolov7-face.txt')
    # parser.add_argument('--save-folder', type=str, default='/root/jinyfeng/datas/sensoro/sensoro_quality_face_alignface_yolov7face')
    
    # parser.add_argument('--img-folder', type=str, default='/data3/ossdata/mz/dy_down_faceimgs')
    # parser.add_argument('--facedetinfo-path', type=str, default='/data3/ossdata/mz/dy_down_yolov7-faceinfo_filter3.txt')
    
    parser.add_argument('--img-folder', type=str, default='/data3/ossdata/mz/wb_down_faceimgs')
    parser.add_argument('--facedetinfo-path', type=str, default='/data3/ossdata/mz/wb_down_yolov7-faceinfo_filter3.txt')
    
    # parser.add_argument('--img-folder', type=str, default='/data3/ossdata/mz/xhs_down_faceimgs')
    # parser.add_argument('--facedetinfo-path', type=str, default='/data3/ossdata/mz/xhs_down_yolov7-faceinfo_filter3.txt')
    
    parser.add_argument('--save-folder', type=str, default='/data2/ossdata/mz/dy_wb_xhs_alignface_filter3')

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
    for info_line in infolists:
#         print(info_line)
        faceinfo = info_line.split('\n')[0]
        filename = faceinfo.split(',')[0]
        ext_flag = filename.endswith(".jpg") or filename.endswith('.jpeg')
        if not ext_flag:
            continue
        idx+=1
        if idx%100 == 0:
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
        
        align_face = face_align_landmark(img, ldmarks)
        savepath = os.path.join(savefolder, filename)
#         print('savepath===========', savepath)
        cv2.imwrite(savepath, align_face)
    
    print('file_num=========', idx)
    
        
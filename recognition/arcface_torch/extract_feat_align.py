import argparse

import os
import cv2
import numpy as np
import torch
from skimage import transform

from backbones import get_model


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
    if len(ndimage.shape) == 2:
        ndimage = np.stack([ndimage, ndimage, ndimage], -1)
    else:
        ndimage = cv2.cvtColor(ndimage, cv2.COLOR_BGR2RGB)
    return ndimage



@torch.no_grad()
def extrace_features(weight, model_name, faceinfo_path, img_folder):
    if img_folder is None or faceinfo_path is None:
        print('img_folder or faceinfo_path is None')
        return
    
    faceinfo_file = open(faceinfo_path, 'r')
    infolists=faceinfo_file.readlines()
    print('len(infolists)=======', len(infolists))
    file_num=0
    features = []
    net = get_model(model_name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    for faceinfo in infolists:
#         print('faceinfo=========', faceinfo)
        faceinfo = faceinfo.split('\n')[0]
        filename = faceinfo.split(',')[0]
        ext_flag = filename.endswith(".jpg")
        if not ext_flag:
            continue
        file_num+=1
        img_path = os.path.join(img_folder, filename)
        img = cv2.imread(img_path)
        print('file_num, filename, img.shape============', file_num, filename, img.shape)
        bbox = faceinfo.split(',')[1:6]
        ldmarks = faceinfo.split(',')[6:]
#         print('bbox, ldmarks===========',bbox, ldmarks)
        ldmarks = np.array(ldmarks).astype(int)
        ldmarks = ldmarks.reshape((5, 2))
#         print('ldmarks===========', ldmarks)
        img = face_align_landmark(img, ldmarks)
#         print('img.shape=========', img.shape)
        
#     filelist = os.listdir(img_folder)
#     print(len(filelist))
        
#         img = cv2.resize(img, (112, 112))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        
        feat = net(img).numpy()
#         print('feat.shape=========', feat.shape)
#         print(feat)
#         features.append(output.data.cpu().numpy())
        features.append(feat)
    print('file_num=========', file_num)
    
    return np.vstack(features)


def extract(test_loader, model):
    batch_time = AverageMeter(10)
    model.eval()
    features = []
    with torch.no_grad():
        end = time.time()
        for i, x in enumerate(test_loader):
            # compute output
            output = model(x)
            features.append(output.data.cpu().numpy())
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

def write_feat(ofn, features):
    print('save features to', ofn)
    features.tofile(ofn)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img-folder', type=str, default=None)
    parser.add_argument('--faceinfo-path', type=str, default=None)
    parser.add_argument('--output-path', default='feat_name.bin', type=str)
    args = parser.parse_args()
    
    features = extrace_features(args.weight, args.network, args.faceinfo_path, args.img_folder)
    #assert features.shape[1] == 256
    
    print('saving extracted features to {}'.format(args.output_path))
    folder = os.path.dirname(args.output_path)
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder)
    if args.output_path.endswith('.bin'):
        write_feat(args.output_path, features)
    elif args.output_path.endswith('.npy'):
        np.save(args.output_path, features)
    else:
        np.savez(args.output_path, features)
    
    
    
    
    
    
    
    

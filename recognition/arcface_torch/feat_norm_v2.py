import argparse

import os
import cv2
import numpy as np
import torch

from backbones import get_model
    
@torch.no_grad()
def extrace_features(weight, network, img_folder, savefolder):
    
    if img_folder is None:
        print('img_folder is None')
        return
    
    filelist = os.listdir(img_folder)
    print(len(filelist))
    file_num=0
    features = []
    net = get_model(network, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    blur_scores_list = []
    print('saving result to {}'.format(savefolder))
#     savefolder = os.path.dirname(args.output_path)
    if savefolder != '' and not os.path.exists(savefolder):
        os.makedirs(savefolder)
        
    for filename in filelist:
        ext_flag = filename.endswith(".jpg")
        if not ext_flag:
            continue
            
        file_num+=1
        if file_num%1000==0:
            print('extract {} images'.format(file_num))
        img_path = os.path.join(img_folder, filename)
        img = cv2.imread(img_path)
        if img.shape[1] != 112:
            print('img.shape[1] != 112====================')
            img = cv2.resize(img, (112, 112))
#         print(file_num, filename, img.shape)
        frame = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        
        feat = net(img).numpy()
#         print('feat.shape==========', feat.shape)
#         print(feat)
        feat_norm = feat/np.linalg.norm(feat, axis=1).reshape(-1, 1)
        feat_norm_sum = np.sum(feat_norm)
#         print('feat_norm_sum=========', feat_norm_sum)
        blur_label = f"{feat_norm_sum:.3f}"
        
        save_filepath = os.path.join(savefolder, blur_label+'_'+filename)
        cv2.imwrite(save_filepath, frame)


        blur_label = float(blur_label)
#         print(type(blur_label))
        blur_scores_list.append(blur_label)
        
    list_max = np.max(blur_scores_list)
    list_min = np.min(blur_scores_list)
    list_mean = np.mean(blur_scores_list)
    blur_scores_list.sort()
    list_mid = blur_scores_list[len(blur_scores_list)//2]
    print('list_min, list_mean, list_mid, list_max===========', list_min, list_mean, list_mid, list_max)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img-folder', type=str, default=None)
    parser.add_argument('--output-path', default='feat_norm', type=str)
    args = parser.parse_args()
    
    
    extrace_features(args.weight, args.network, args.img_folder, args.output_path)
    
    
    
    
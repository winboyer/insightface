import argparse

import os
import cv2
import numpy as np
import torch

from backbones import get_model
import psutil
import time

@torch.no_grad()
def extrace_features(net, name, img_folder):
    if img_folder is None:
        print('img_folder is None')
        return
    
    txtfile = open(img_folder+'.txt', 'r')
    facelists = txtfile.readlines()
    txtfile.close()
    # filelist = os.listdir(img_folder)
    print('len(facelists)============', len(facelists))
    file_num=0
    features = []
    
    # p = psutil.Process()
    # p.cpu_affinity([0,1,2,3,4,5,6,7,8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('device========', device)
    net = net.cuda()
    for filename in facelists:
        filename = filename.split('\n')[0]
        # print('filename=======', filename)
        ext_flag = filename.endswith(".jpg") or filename.endswith(".jpeg")
        if not ext_flag:
            continue
            
        file_num+=1
        if file_num%100==0:
            print('extract {} images'.format(file_num))
        img_path = os.path.join(img_folder, filename)
        img = cv2.imread(img_path)
#         print(file_num, filename, img.shape)
        if img.shape[1] != 112:
            print('img.shape[1] != 112====================')
            img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        
        # img = torch.from_numpy(img).unsqueeze(0).float()
        img = torch.from_numpy(img).unsqueeze(0).cuda()
        # print(img)
        # img.div_(255).sub_(0.5).div_(0.5)
        img=torch.div(img, 255)
        img=torch.sub(img, 0.5)
        img=torch.div(img, 0.5)
        # print(img)
        # net = net.cuda()
        feat = net(img)
        feat = feat.cpu().numpy()
        # print('feat.shape==========', feat.shape)
        # print(feat)
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
    parser.add_argument('--output-path', default='feat_name.bin', type=str)
    args = parser.parse_args()
    
    net = get_model(args.network, fp16=False)
    net.load_state_dict(torch.load(args.weight))
    net.eval()

    features = extrace_features(net, args.network, args.img_folder)
    print(features.shape)
    assert features.shape[1] == 256
    
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
    
    
    
    
    
    
    
    

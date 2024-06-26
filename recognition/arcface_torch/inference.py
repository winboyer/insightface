import argparse

import cv2
import numpy as np
import torch

from backbones import get_model


@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        print('img is None')
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).numpy()
    feat_sum = np.sum(feat)
    feat_norm = feat/np.linalg.norm(feat, axis=1).reshape(-1, 1)
    feat_norm_sum = np.sum(feat_norm)
    print('feat, feat_norm, feat_sum, feat_norm_sum=========',feat, feat_norm, feat_sum, feat_norm_sum)
#     absfeat = np.absolute(feat)
#     absfeat_sum = np.sum(absfeat)
#     absfeat_norm = absfeat/np.linalg.norm(absfeat, axis=1).reshape(-1, 1)
#     absfeat_norm_sum = np.sum(absfeat_norm)
#     print('absfeat, absfeat_norm, absfeat_sum, absfeat_norm_sum===========', absfeat, absfeat_norm, absfeat_sum, absfeat_norm_sum)
#     print(feat, feat.shape, feat_sum, absfeat_sum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    inference(args.weight, args.network, args.img)

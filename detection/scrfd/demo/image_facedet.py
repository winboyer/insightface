from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import cv2
import numpy as np

def main():
    parser = ArgumentParser()
#     parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
#     parser.add_argument(
#         '--score-thr', type=float, default=0.8, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    image_path = '/root/jinyfeng/datas/sensoro/sensoro_person_info/nh_622326.jpg'
    
    result = inference_detector(model, image_path)
    
#     image = cv2.imread(args.img)
    image = cv2.imread(image_path)
    
    bbox_result=np.vstack(result)
#     print('len(bbox_result)=========', len(bbox_result))
#     print('bbox_result=========', bbox_result)
    for bbox in bbox_result:
        print(bbox)
    
    inds = np.where(bbox_result[:, -1] >= args.score_thr)[0]
    print('len(inds)=========', len(inds))
    print('inds=========', inds)
    
#     # show the results
    for idx in inds:
        bbox_i = bbox_result[idx]
        score = bbox_i[-1]
        bbox_i = bbox_i.astype(int)
        print(idx, bbox_i, score)
        cv2.rectangle(image, (bbox_i[0], bbox_i[1]), (bbox_i[2], bbox_i[3]), (0,255,0), 2)
    
#     cv2.imwrite('./test_result.jpg', image)
    

if __name__ == '__main__':
    main()

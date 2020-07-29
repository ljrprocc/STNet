import numpy as np
import sys
sys.path.append('/home/jingru.ljr/Motif-Removal/')
from utils.pypse import pse as pypse
import cv2
import tqdm

min_area = 50
min_score = 0.95

def mask2bboxes(score):
    pred = pypse([score], min_area)
    label_nums = np.max(pred) + 1
    bboxes = []
    for i in range(1, label_nums):
        points = np.array(np.where(pred == i)).transpose((1,0))[:, ::-1]

        if points.shape[0] < min_area:
            continue
        
        score_i = np.mean(score[pred == i])
        if score_i < min_score:
            continue

        rect = cv2.minAreaRect(points)
        bbox = cv2.boxPoints(rect)
        # bbox = bbox.astype('int32')
        bboxes.append(bbox)

    return bboxes

def run_boxes(text_boxes, pred_mask, write_path, write_res_name):
    # print(text_boxes.dtype, type(text_boxes))
    cv2.imwrite('/home/jingru.ljr/checkpoints/c.jpg', text_boxes)
    imgk = cv2.imread('/home/jingru.ljr/checkpoints/c.jpg')
    # print(imgk[:, :, 0] - text_boxes[:, :, 1])
    logit = np.log((pred_mask) / (1 - pred_mask + 1e-9) + 1e-9)
    output = (np.sign(logit - 1) + 1) / 2
    kernel_mask = output * logit
    # print(np.max(kernel_mask))
    bboxes = mask2bboxes(kernel_mask.astype(np.uint8))
    for bbox in bboxes:
        cv2.drawContours(imgk, [bbox.reshape(4,2).astype(np.int32)], -1, (0, 255, 0), 2)
    cv2.imwrite(write_path, imgk)
    # exit(-1)
    fwrite = open(write_res_name, 'w')
    fwrite.close()
    np.savetxt(write_res_name, X=np.array(bboxes).reshape(-1, 8), delimiter=',', fmt='%.2f')


def visualization(img_path, write_path):
    ori_img = cv2.imread(img_path)
    imgs = np.split(ori_img[5:], 5, axis=0)
    img, pred_mask= imgs[0][:-5, 5:-5, :], imgs[3][:-5, 5:-5, 0] /255
    logit = np.log((pred_mask) / (1 - pred_mask + 1e-9) + 1e-9)
    output = (np.sign(logit - 1) + 1) / 2
    kernel_mask = output * logit
    # print(np.max(kernel_mask))
    bboxes = mask2bboxes(kernel_mask.astype(np.uint8))
    text_boxes = img.copy()
    for bbox in bboxes:
        cv2.drawContours(text_boxes, [bbox.reshape(4,2).astype(np.int32)], -1, (0, 255, 0), 2)
    cv2.imwrite(write_path, text_boxes)
    return bboxes


def single_debug(img_dir):
    import os
    write_path = '/home/jingru.ljr/checkpoints/result/'
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    train_tag = 'icdar_total_256'
    if not os.path.exists(os.path.join(write_path, train_tag)):
        os.mkdir(os.path.join(write_path, train_tag))

    vis_path = os.path.join(write_path, train_tag, 'vis/')
    res_path = os.path.join(write_path, train_tag, 'res/')
    
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)
    
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    
    for img_name in tqdm.tqdm(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        write_names = os.path.join(vis_path, img_name)
        write_res_name = os.path.join(res_path, img_name + '.txt')
        bboxes = visualization(img_path, write_names)
        fwrite = open(write_res_name, 'w')
        fwrite.close()
        np.savetxt(write_res_name, X=np.array(bboxes).reshape(-1, 8), delimiter=',', fmt='%.2f')


if __name__ == '__main__':
    img_dir = '/home/jingru.ljr/checkpoints/icdar_total2/images/'
    single_debug(img_dir)

    
    
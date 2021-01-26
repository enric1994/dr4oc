from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import json
import random
from tqdm import tqdm

dir_name = os.path.dirname(os.path.abspath(__file__))

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def generate_data(im_path, labels, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    points = np.array(labels).astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def main(input_dataset_path, output_dataset_path, min_size=512, max_size=2048, size_x=1024, size_y=768):

    labels = []
    image_paths =[]
    labels_files = [x for x in os.listdir(input_dataset_path) if '.json' in x]

    for l in tqdm(labels_files):
        with open(os.path.join(input_dataset_path, l)) as f:
            data=json.load(f)
        
        image_name = data['global']['scene_name']
        image_path = os.path.join(input_dataset_path, image_name + '.png')
        
        # remove points outisde image
        points = [x[0][:2] for x in data['response']['vertexs'] if x[0][0] in range(0,size_x) and  x[0][1] in range(0,size_y)]
        #3
        if len(points) > 0:
            image_paths.append(image_path)
            labels.append(points)

    dataset = list(zip(image_paths, labels))
    random.shuffle(dataset)

    datasets = {
        'train': dataset
    }
    

    for phase in ['train']:
        if not os.path.exists(os.path.join(output_dataset_path, phase)):
            os.makedirs((os.path.join(output_dataset_path, phase)))
        for im_path, labels in tqdm(datasets[phase]):
            try:
                name = im_path.split('/')[5]
                im, points = generate_data(im_path, labels, min_size, max_size)
                im_save_path = os.path.join(output_dataset_path, phase, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('png', 'npy')
                np.save(gd_save_path, points)
            except:
                print('error!',im_path)

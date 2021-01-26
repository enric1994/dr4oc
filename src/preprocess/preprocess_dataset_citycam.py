from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import json
import random
from tqdm import tqdm
from xml.dom import minidom


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


def main(input_dataset_path, output_dataset_path, min_size=512, max_size=2048, size_x=762, size_y=512):

    
    datasets = {}
    for split in ['train', 'test']:
        labels = []
        images =[]
        
        images_paths = '{}/train_test_separation/Downtown_{}.txt'.format(input_dataset_path, split.title())
        with open(images_paths) as f:
            data = f.readlines()
            data = [x.strip() for x in data]
            
            for d in tqdm(data):
                image_paths = os.listdir(input_dataset_path + '/' + d.split('-')[0] + '/' + d)
                image_paths = [x for x in image_paths if '.xml' in x]
                for i in image_paths:
                    centroids = []

                    label_path = input_dataset_path + '/' +  d.split('-')[0] + '/' + d + '/' + i
                    try:
                        xmldoc = minidom.parse(label_path)
                        
                        vehicles = xmldoc.getElementsByTagName('vehicle')
                        for vehicle in vehicles:
                            bndbox = vehicle.getElementsByTagName("bndbox")[0]
                            xmax = int(bndbox.getElementsByTagName("xmax")[0].childNodes[0].nodeValue)
                            xmin = int(bndbox.getElementsByTagName("xmin")[0].childNodes[0].nodeValue)
                            ymax = int(bndbox.getElementsByTagName("ymax")[0].childNodes[0].nodeValue)
                            ymin = int(bndbox.getElementsByTagName("xmin")[0].childNodes[0].nodeValue)

                            centroid_x = ( xmax + xmin ) / 2
                            centroid_y = ( ymax + ymin ) / 2
                            
                            centroids.append([centroid_x,centroid_y])

                        
                        labels.append(centroids)    
                        images.append(label_path.replace('.xml', '.jpg'))
    
                    except:
                        # Malformed XML
                        pass

        dataset = list(zip(images, labels))
        datasets[split] = dataset

    for phase in ['train', 'test']:
        if not os.path.exists(os.path.join(output_dataset_path, phase)):
            os.makedirs((os.path.join(output_dataset_path, phase)))
        for im_path, labels in tqdm(datasets[phase]):
            try:
                name = im_path.split('/')[5]
                im, points = generate_data(im_path, labels, min_size, max_size)
                im_save_path = os.path.join(output_dataset_path, phase, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)
            except:
                print('error!',im_path)

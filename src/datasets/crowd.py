from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import scipy.io as sio
import json
from tqdm import tqdm
import cv2


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    assert np.sum(discrete_map) == num_gt
    return discrete_map


class Base(data.Dataset):
    def __init__(self, root_path, crop_size, downsample_ratio=8):

        self.root_path = root_path
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i] #, 0]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), st_size, torch.from_numpy(
            gt_discrete.copy()).float()


class Crowd_qnrf(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))
        if method not in ['train', 'val']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name

# class Crowd_blendercam(Base):
#     def __init__(self, root_path, crop_size,
#                  downsample_ratio=8,
#                  method='train'):
#         super().__init__(root_path, crop_size, downsample_ratio)
#         self.method = method
#         self.im_list = sorted(glob(os.path.join(self.root_path, '*.png')))
#         print('number of img: {}'.format(len(self.im_list)))
#         if method not in ['train', 'val']:
#             raise Exception("not implement")

#     def __len__(self):
#         return len(self.im_list)

#     def __getitem__(self, item):
#         img_path = self.im_list[item]
#         gd_path = img_path.replace('png', 'npy')
#         img = Image.open(img_path).convert('RGB')
#         if self.method == 'train':
#             keypoints = np.load(gd_path)
#             return self.train_transform(img, keypoints)
#         elif self.method == 'val':
#             keypoints = np.load(gd_path)
#             img = self.trans(img)
#             name = os.path.basename(img_path).split('.')[0]
#             return img, len(keypoints), name

# class Crowd_penguins(Base):
#     def __init__(self, root_path, crop_size,
#                  downsample_ratio=8,
#                  method='train'):
#         super().__init__(root_path, crop_size, downsample_ratio)
#         self.method = method
#         self.im_list = sorted(glob(os.path.join(self.root_path, '*.JPG')))
#         print('number of img: {}'.format(len(self.im_list)))
#         if method not in ['train', 'val']:
#             raise Exception("not implement")

#     def __len__(self):
#         return len(self.im_list)

#     def __getitem__(self, item):
#         img_path = self.im_list[item]
#         gd_path = img_path.replace('JPG', 'npy')
#         img = Image.open(img_path).convert('RGB')
#         if self.method == 'train':
#             keypoints = np.load(gd_path)
#             return self.train_transform(img, keypoints)
#         elif self.method == 'val':
#             keypoints = np.load(gd_path)
#             img = self.trans(img)
#             name = os.path.basename(img_path).split('.')[0]
#             return img, len(keypoints), name


class Crowd_nwpu(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))

        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name
        elif self.method == 'test':
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, name


class Crowd_sh(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'test']:
            raise Exception("not implement")
        self.im_list = sorted(glob(os.path.join(self.root_path, self.method, 'images', '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        # print(img_path)
        name = os.path.basename(img_path).split('.')[0]
        gd_path = os.path.join(self.root_path, self.method, 'ground_truth', 'GT_{}.mat'.format(name))
        img = Image.open(img_path).convert('RGB')
        keypoints = sio.loadmat(gd_path)['image_info'][0][0][0][0][0]
        

        # if self.method == 'train':
        #     return self.train_transform(img, keypoints)
        # elif self.method == 'test':
        img = self.trans(img)
        return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), st_size, torch.from_numpy(
            gt_discrete.copy()).float()

class Crowd_penguins(Base):
    def __init__(self, root_path, crop_size, limit,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)

        # self.trans = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     transforms.Resize((1024,768))
        # ])
        self.limit = int(limit)
        self.c = 0
        self.method = method
        if method not in ['train', 'val','test']:
            raise Exception("not implement")
        
        # self.im_list = sorted(glob(os.path.join(self.root_path, 'images', '*.jpg')))

        # datasets = {}

        # for split in ['train','val','test']:

        self.keypoints = []
        self.im_list =[]

        splits_path = os.path.join(root_path, 'Splits_2016_07_11', 'imdb.json')
        with open(splits_path) as f:
            splits = json.load(f)
            image_paths = [ os.path.join(root_path, x) for x in splits['imdb'][method]]
        
        annotations = {}
        annotations_folder = os.path.join(root_path, 'CompleteAnnotations_2016-07-11')
        annotation_names = [x[:-5] for x in os.listdir(annotations_folder) if '.json' in x ]
        
        for annotation_name in tqdm(annotation_names):

            with open(os.path.join(annotations_folder,annotation_name + '.json')) as f:
                annotations[annotation_name] = json.load(f)
        
        random.shuffle(image_paths)

        for i, image_path in tqdm(enumerate(image_paths)):
            group_name, image_name = image_path.split('/')[-2:]
            image_index = int(image_name.split('_')[1].split('.')[0])
            locations = annotations[group_name]['dots'][image_index - 1]['xy']
            clean = True
            
            # if self.c >= self.limit:
            #     break

            if locations != None and locations != '_NaN_':
                counts = [len(x) for x in locations if x != '_NaN_' and x != None]

                # Get max count
                if len(counts) > 0:
                    # most_common = max(set(counts), key=counts.count)
                    # counts_index = counts.index(most_common)
                    counts_index = counts.index(max(counts))

                    if locations[counts_index] != '_Nan_' and locations[counts_index] != None:
                        if  len(locations[counts_index]) > 0:

                            for i in locations[counts_index]:
                                # print(i)
                                if not isinstance(i, list):
                                    clean = False
                                else:
                                    clean = True
                                    # print(i)
                                    # import pdb;pdb.set_trace()
                                
                            if clean:
                                self.im_list.append(image_path)
                                self.keypoints.append(np.array(locations[counts_index]))
                                self.c+=1
                                                
            # dataset = list(zip(images, labels))

        #######


        print('number of img: {}'.format(len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        keypoints = self.keypoints[item]
        # print(keypoints)
        # print(img_path)
        name = os.path.basename(img_path).split('.')[0]
        # gd_path = os.path.join(self.root_path, 'ground_truth', 'GT_{}.mat'.format(name))
        img = Image.open(img_path).convert('RGB')
        wd, ht = img.size

        # img = img.resize((wd//2, ht//2), Image.BICUBIC)
        # keypoints = keypoints * 0.5

        # keypoints = sio.loadmat(gd_path)['image_info'][0][0][0][0][0]

        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'val' or self.method == 'test':
            img = self.trans(img)
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        # if len(keypoints) == 1:
        #     keypoints = []
        #     keypoints.append(keypoints)
        # print(keypoints)
        # img = img.resize((1024, 768), Image.BICUBIC)
        # keypoints = keypoints * 0.5
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # print(wd,ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
            # print(keypoints)
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        # print(img.size)
        img = F.crop(img, i, j, h, w)
        
        if len(keypoints) > 0:
            # print(keypoints)
            keypoints = keypoints - [j, i]
            # print(keypoints)
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), st_size, torch.from_numpy(
            gt_discrete.copy()).float()


class Crowd_trancos(Base):
    def __init__(self, root_path, crop_size, limit,
                 downsample_ratio=8,
                 method='training'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.limit = int(limit)
        
        splits_path = os.path.join(root_path, 'image_sets', self.method + '.txt')

        with open(splits_path) as f:
            self.im_list = [ os.path.join(root_path, 'images', x)[:-1] for x in f]
            self.im_list = self.im_list[:self.limit]


            
        print('number of img: {}'.format(len(self.im_list)))


        #
        label_path = self.im_list[0].split('.')[0] + '.txt'
        # keypoints = sio.loadmat(gd_path)['image_info'][0][0][0][0][0]
        

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        # print(img_path)
        name = os.path.basename(img_path).split('.')[0]
        # gd_path = os.path.join(self.root_path, self.method, 'ground_truth', 'GT_{}.mat'.format(name))
        img = Image.open(img_path).convert('RGB')

        label_path = img_path.split('.')[0] + '.txt'
        # keypoints = sio.loadmat(gd_path)['image_info'][0][0][0][0][0]
        keypoints = []
        with open(label_path) as f:
            for raw_keypoint in f:
                keypoint = raw_keypoint.replace('\t','.').replace('\n','').split('.')
                keypoints.append([int(keypoint[0]), int(keypoint[1])])
        
        keypoints = np.array(keypoints)

        # wd, ht = img.size
        # img = img.resize((wd//2, ht//2), Image.BICUBIC)
        # keypoints = keypoints * 0.5

        if self.method == 'training':
            return self.train_transform(img, keypoints)
        elif self.method == 'validation' or self.method == 'test':
            img = self.trans(img)
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), st_size, torch.from_numpy(
            gt_discrete.copy()).float()

class Crowd_blendercam(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = []
        self.keypoints = []
        self.image_paths = sorted(glob(os.path.join(self.root_path, '*.png')))
        
        self.label_paths = [x.replace('png','json') for x in self.image_paths]

        self.size_x, self.size_y = Image.open(self.image_paths[0]).convert('RGB').size
        


        for l in tqdm(self.label_paths):
            try:
                with open(os.path.join(self.root_path, l)) as f:
                    data=json.load(f)
                
                    # remove points outisde image
                    points = [x[0][:2] for x in data['response']['vertexs'] if x[0][0] in range(0,self.size_x) and  x[0][1] in range(0,self.size_y)]
                
                    if len(points) > 0:
                        self.im_list.append(l.replace('json','png'))
                        self.keypoints.append(points)
            except:
                print('Missing:', l)

        print('number of img: {}'.format(len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        # print(img_path)
        name = os.path.basename(img_path).split('.')[0]
        # gd_path = os.path.join(self.root_path, self.method, 'ground_truth', 'GT_{}.mat'.format(name))
        img = Image.open(img_path).convert('RGB')
        keypoints = np.array(self.keypoints[item])

        # check trancos!!
        # img = img.resize((self.size_x//2, self.size_y//2), Image.BICUBIC)
        # keypoints = keypoints * 0.5
        

        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'test':
            img = self.trans(img)
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), st_size, torch.from_numpy(
            gt_discrete.copy()).float()



class Crowd_apples(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method

        self.im_list = []
        self.keypoints = []

        self.image_paths = sorted(glob(os.path.join(self.root_path,  'train' ,'images', '*.png')))

        # random.shuffle(self.image_paths)

        if self.method=='train':
            self.image_paths=self.image_paths[:len(self.image_paths)//2]
        elif self.method=='test':
            self.image_paths=self.image_paths[len(self.image_paths)//2:]
        else:
            raise Exception("not implement")

        for img_path in tqdm(self.image_paths):
            mask_path = img_path.replace('images','masks')
            mask = cv2.imread(mask_path)
            if mask is not None:
                mask_channel = mask[:,:,0]
                total = len(np.unique(mask_channel)) 
                local_keypoints = []
                for k in range(1,total):
                    coords = np.where(mask_channel == k)
                    # import pdb;pdb.set_trace()
                    if len(coords[0]) > 0:
                        x,y = coords[0][0], coords[1][0]
                        local_keypoints.append([x,y])

                self.im_list.append(img_path)
                self.keypoints.append(local_keypoints)
        
        print('number of img: {}'.format(len(self.im_list)))
            
        
    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        # print(img_path)
        name = os.path.basename(img_path).split('.')[0]
        # gd_path = os.path.join(self.root_path, self.method, 'ground_truth', 'GT_{}.mat'.format(name))
        img = Image.open(img_path).convert('RGB')
        keypoints = np.array(self.keypoints[item])

        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'test':
            img = self.trans(img)
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        # if self.method == 'train':
        #     return self.train_transform(img, keypoints)
        # elif self.method == 'test':
        img = self.trans(img)
        return img, self.labels[item], name

class Crowd_blendercam_preprocess(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.png')))
        print('number of img: {}'.format(len(self.im_list)))
        if method not in ['train', 'val']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('png', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name

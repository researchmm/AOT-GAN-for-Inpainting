import os
import io
import scipy
import torch
import random
import time 
import lmdb 
import math 
import argparse
import pickle 
import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import canny
from xml.etree import ElementTree as ET

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader



def random_bbox(img_width, img_height):
    """Generate a random tlhw regular mask 
    """
    vertical_margin = horizontal_margin = 0
    mask_height = img_height//2 
    mask_width = img_width//2
    max_delta_height = img_height // 8
    max_delta_width = img_width // 8
    maxt = img_height - vertical_margin - mask_height
    maxl = img_width - horizontal_margin - mask_width
    mask = np.zeros((img_height, img_width), np.uint8)

    t = np.random.randint(vertical_margin, maxt)
    l = np.random.randint(horizontal_margin, maxl)
    h = np.random.randint(max_delta_height//2+1)
    w = np.random.randint(max_delta_width//2+1)
    mask[t+h:t+mask_height-h,
         l+w:l+mask_width-w] = 1
    return mask


def center_bbox(img_width, img_height):
    """Generate a center square mask 
    """
    mask = np.zeros((img_height, img_width), np.uint8)
    mask[:, img_height//4:img_height//4*3,
         img_width//4:img_width//4*3] = 1
    return mask


def random_stroke(img_width, img_height):
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 40
    average_radius = math.sqrt(img_height*img_height+img_width*img_width) / 8
    mask = Image.new('L', (img_width, img_height), 0)

    steps = 6
    for _ in range(np.random.randint(1, steps)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(
                    2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)),
                        int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius,
                                  scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                          fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    return mask


def bbox2np(img_width, img_height, bbox, pad_rate=0):
    mask = np.zeros((img_height, img_width), np.uint8)
    for (xmin, ymin, xmax, ymax) in bbox:
        pad = int(pad_rate*min(xmax-xmin, ymax-ymin))
        mask[max(0,ymin-pad):min(ymax+pad, img_height), 
             max(0,xmin-pad):min(xmax+pad, img_width)] = 1
    return Image.fromarray(mask*255)


def outside_xml(oriw, orih, bbox):
    if bbox is None:
        return random_bbox(oriw, orih)
    
    mask = np.zeros((orih, oriw))
    logo_mask = np.array(bbox2np(oriw, orih, bbox))//255
    random_mask = small_block(oriw, orih)
    
    if np.sum(logo_mask) < 0.6*oriw*orih:
        mask = ((mask + random_mask)>0) * (1-logo_mask)
        iters = 5 
        while np.sum(mask) < 0.1*oriw*orih and iters>0:
            random_mask = small_block(oriw, orih)
            mask = ((mask+random_mask)>0) * (1-logo_mask)
            iters -= 1
    else:
        mask = small_block(oriw, orih)
    return np.array(mask).astype(np.uint8)
    

def small_block(width, height, nums=3):
    mask = np.zeros((height, width)).astype(np.uint8)
    margin_width = width//8
    margin_height = height//8
    for i in range(nums):
        x = random.randint(margin_width, width - margin_width)
        y = random.randint(margin_height, height - margin_height)
        w = random.randint(margin_width, width//2)
        h = random.randint(margin_height, height//2)
        mask[y:min(y+h, height), x:min(x+w, width)] = 1
    return mask


    
class LmdbReader(object):
    lmdb_env = None

    def __init__(self):
        super(LmdbReader, self).__init__()

    @staticmethod
    def build_lmdb_env(lmdb_path):
        if LmdbReader.lmdb_env is None:
            LmdbReader.lmdb_env = lmdb.open(lmdb_path, max_readers=64, readonly=True,
                                            lock=False, readahead=False, meminit=False,)
        return LmdbReader.lmdb_env

    @staticmethod
    def read(path, key, val_type='img'):
        env = LmdbReader.build_lmdb_env(path)
        with env.begin(write=False) as txn:
            try: 
                if val_type == 'int':
                    val = txn.get(key).decode('utf-8')
                    val = int(val)
                elif val_type == 'img':
                        val = io.BytesIO(txn.get(key))
                        val = Image.open(val)
                elif val_type == 'list':
                    val = io.BytesIO(txn.get(key))
                    val = pickle.load(val)
                    val = list(val)
            except: 
                val = None
        return val

    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, size, mask_type='bbox', split='train'):
        super(Dataset, self).__init__()
        self.lmdb_path = path
        self.mask_type = mask_type
        self.split = split
        self.size = size
        self.train_total = LmdbReader.read(path, 'train-total'.encode('utf-8'), val_type='int')
        self.test_total = LmdbReader.read(path, 'test-total'.encode('utf-8'), val_type='int')
        
        self._train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]
        )

        self._test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]
        )


    def set_split(self, split='train'):
        if split == 'train':
            self.split = split 
        else:
            self.split = 'test'

            
    def __len__(self):
        if self.split == 'train':
            return self.train_total
        else:
            return self.test_total
  

    def set_subset(self, start, end):
        self.data = self.data[start:end] 

        
    def __getitem__(self, index):
        key = f'{self.split}-{str(index).zfill(7)}-image'
        orig_img = LmdbReader.read(self.lmdb_path, key.encode('utf-8'), val_type='img')
        if self.split == 'train':
            # obtain mask
            if self.mask_type != 'xml':
                img = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.Resize(self.size),
                                          transforms.RandomCrop(self.size),])(orig_img)
                if self.mask_type == 'bbox':
                    mask = random_bbox(self.size, self.size)
                elif self.mask_type == 'stroke':
                    mask = random_stroke(self.size, self.size)
                elif self.mask_type == 'comp':
                    mask = np.array(np.logical_or(random_bbox(self.size, self.size),
                                                  random_stroke(self.size, self.size))).astype(np.uint8)
                mask = Image.fromarray(mask*255)
            else: # self.mask_type == 'xml':
                oriw, orih = orig_img.size
                label_name = f'{self.split}-{str(index).zfill(7)}-label'
                bbox = LmdbReader.read(self.lmdb_path, label_name.encode('utf-8'), val_type='list')
                mask = Image.fromarray(outside_xml(oriw, orih, bbox)*255)

                # resize 
                rate = 512.0 / min(orih, oriw)
                neww, newh = int(oriw*rate), int(orih*rate)
                img = orig_img.resize((neww, newh), Image.BILINEAR)
                mask = mask.resize((neww, newh), Image.NEAREST)
            
                x = random.randint(0, np.maximum(0, img.size[0] - self.size))
                y = random.randint(0, np.maximum(0, img.size[1] - self.size))
                img = img.crop((x, y, x+self.size, y+self.size))
                mask = mask.crop((x, y, x+self.size, y+self.size))
                
                if np.random.normal() > 0:
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    
                if np.sum(np.array(mask)) == 0:
                    mask = Image.fromarray(small_block(self.size, self.size)*255)
                
            img = self._train_transform(img)
            return key, img, transforms.ToTensor()(mask)
        else:
            oriw, orih = orig_img.size
            rate = (self.size + 0.0) / max(orih, oriw)
            neww, newh = int(oriw*rate), int(orih*rate)
            img = orig_img.resize((neww, newh), Image.BILINEAR)
            
            label_name = f'{self.split}-{str(index).zfill(7)}-label'
            bbox = LmdbReader.read(self.lmdb_path, label_name.encode('utf-8'), val_type='list')
            if bbox is not None: 
                orig_mask = bbox2np(oriw, orih, bbox)
                mask = orig_mask.resize((neww, newh), Image.NEAREST)
            else:
                mask = random_bbox(neww, newh)
                mask = Image.fromarray(mask*255)
            
            img = F.pad(img, (0, 0, max(512 - neww, 0), max(512 - newh, 0)), fill=0, padding_mode='reflect')
            mask = F.pad(mask, (0, 0, max(512-neww, 0), max(512 - newh, 0)), fill=0, padding_mode='reflect')
            
            orig_img = self._test_transform(orig_img)
            img = self._test_transform(img)
            mask = transforms.ToTensor()(mask)
            return key, orig_img, img, mask
            
    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self, batch_size=batch_size, drop_last=True)
            for item in sample_loader:
                yield item

                
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser(description="ca")
    parser.add_argument('--lmdb_path', type=str, default='/data07/t-yazen/lsun_data/logos')
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()
    
    
    d = Dataset(args.lmdb_path, args.size)
    d.set_split('test')
    print(len(d), ' for testing')
    for i in range(5): 
        key, orig_img, img, mask = d[i]
        print(orig_img.size(), np.unique(mask.numpy()))
        orig_img = (orig_img.permute(1,2,0).numpy()+1)/2*255
        orig_img = Image.fromarray(orig_img.astype(np.uint8))
        
        oriw, orih = orig_img.size
        rate = (args.size+0.0) / max(orih, oriw)
        neww, newh = int(oriw*rate), int(orih*rate)
        
        img = img*(1.-mask) + mask
        img = (img.permute(1,2,0).numpy()+1)/2*255
        img = Image.fromarray(img.astype(np.uint8))
        mask = mask.squeeze().numpy()
        mask = Image.fromarray(mask.astype(np.uint8))
        img = F.crop(img, 0, 0, newh, neww)
        mask = F.crop(mask, 0, 0, newh, neww)
        print(oriw, orih, neww, newh)
        mask = np.expand_dims(mask, axis=-1)
        img = np.array(np.array(img)*(1.-mask) + mask*255).astype(np.uint8)
        Image.fromarray(img).save(f'comp-{key}.png')
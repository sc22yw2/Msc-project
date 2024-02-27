import torch
import numpy as np
from torchvision import datasets, transforms
import torch.utils.data as data
from PIL import Image
import random
import os
import cv2
import random
from glob import glob
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        raise(RuntimeError('No Module named accimage'))
    else:
        return pil_loader(path)

class ImageNetData(data.Dataset):
    def __init__(self, img_root, img_file,  transform=None, loader=default_loader):
        self.root = img_root

        self.imgs = glob(os.path.join(img_root,img_file,"*.jpg"))

        
        self.transform = transform
        self.loader = loader
        self.get_min_max_age()
        age_cls_num_list = {}
        sex_cls_num_list = {}
        for path in self.imgs:
            basename = os.path.basename(path)
            age = basename.split("_")[0]
            sex = basename.split("_")[1]
            if sex == "3":
                print(path)
            if age not in age_cls_num_list:
                age_cls_num_list[age] = 1
            else:
                age_cls_num_list[age] += 1
            if sex not in sex_cls_num_list:
                sex_cls_num_list[sex] = 1
            else:
                sex_cls_num_list[sex] += 1

        if "3" in sex_cls_num_list:
            del sex_cls_num_list["3"]

        self.cls_num = {"age_cls_num_list":list(age_cls_num_list.values()),"sex_cls_num_list":list(sex_cls_num_list.values())}

    def get_min_max_age(self):
        ages = set()
        for path in self.imgs:
            basename = os.path.basename(path)
            age = basename.split("_")[0]
            age = int(age)
            ages.add(age)
        self.min_age = list(ages)[0]
        self.max_age = list(ages)[1]

    def __getitem__(self, index):
        path = self.imgs[index]
        basename = os.path.basename(path)
        age = basename.split("_")[0]
        sex = basename.split("_")[1]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        try:
            age_ = int(age)
            sex_ = int(sex)
        except:
            print(path)
        if sex_ not in [0,1]:
            sex_ = 1
        if age_ >= 110:
            age_ =109



        return img, age_,sex_
    
    def __len__(self):
        return len(self.imgs)


import torch
import os 
from glob import glob 
from torch.utils.data import Dataset
from PIL import Image
import random
from torch.utils.data.sampler import Sampler
import numpy as np
import math 

class Medical_DB(Dataset):
    def __init__(self, folder = '/home/datasets/Medical_dataset', transform = None):

        self.images = glob('/home/datasets/Medical_dataset/cholec80/Images/**/*.jpg', recursive=True)
        self.transform = transform

        self.labels = {}
        count = 0
        for image in self.images:
            if image[:image.rfind('/')] not in self.labels.keys():
                self.labels[image[:image.rfind('/')]] = count
                count +=1


    def __len__(self):
        
        return len(self.images)

    def label_num(self):

        return len(self.labels.keys())


    def __getitem__(self, index):
        
        img = Image.open(self.images[index]).convert('RGB')
        label = self.labels[self.images[index][:self.images[index].rfind('/')]]
        
        if self.transform:
            img = self.transform(img)
        return (img,label)


class Proposed_Medical_DB(Dataset):
    def __init__(self, folder = '/home/datasets/Medical_dataset', transform = None, negative_transform = None):

        self.images = sorted(glob('/home/datasets/Medical_dataset/cholec80/Images/**/*.jpg', recursive=True))
        self.transform = transform
        self.negative_transform = negative_transform

        self.labels = {}
        count = 0
        for image in self.images:
            if image[:image.rfind('/')] not in self.labels.keys():
                self.labels[image[:image.rfind('/')]] = count
                count +=1


    def __len__(self):
        
        return len(self.images)

    def label_num(self):

        return len(self.labels.keys())


    def __getitem__(self, index):
        
        img = Image.open(self.images[index]).convert('RGB')
        label = self.labels[self.images[index][:self.images[index].rfind('/')]]

        num = list(self.labels.values())
        #print(num)
        num.remove((index//5000))
        index_negative = random.choice(num)*5000 +  random.randint(1,4999)
        img_negative = Image.open(self.images[index_negative]).convert('RGB')

        if self.transform:
            img = self.transform(img)
            img_negative = self.negative_transform(img_negative)
        return (img,label, img_negative)





class CustomSampler(Sampler):
    def __init__(self, dataset, batchsize):
        self.length = len(dataset)
        self.num_list = [i for i in range(len(dataset))]
        self.distributed = [self.num_list[i*batchsize: (i+1)*batchsize] for i in range(0,math.ceil(len(self.num_list)/batchsize))]
        
    def __iter__(self):
        random.shuffle(self.distributed)
        return iter(np.array(self.distributed).reshape(self.length).tolist())
    
    def __len__(self):
        return len(self.num_list)
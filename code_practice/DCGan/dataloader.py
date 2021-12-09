from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import glob
import random

class My_data(data.Dataset):
    def __init__(self,path,transform=None):
        self.path=path
        self.img_list=glob.glob(self.path+'/*.jpg')
        self.transform=transform
    
    def __getitem__(self,index):
        img_path=self.img_list[index]
        img=Image.open(img_path)
        return self.transform(img)
    
    def __len__(self):
        return len(self.img_list)
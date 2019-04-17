import numpy as np
import torch 
import matplotlib.pyplot as plt
from .preprocess import *

class CellDataset(torch.utils.data.Dataset):
    def __init__(self, size=512, train=True):
        self.image_path = []
        self.label_path = []
        self.size = size
        self.size = 32
        num = 81
        sequence = np.random.permutation(num)
        sequence = sequence[:int(num* 0.8)] if train else sequence[int(num*0.8):num]
        
        for i in sequence:
            self.image_path.append("./data/image/" + str(i) + ".jpg")
            self.label_path.append("./data/label/" + str(i) + ".jpg")        

    def __getitem__(self, index):
        image = loadImage(self.image_path[index], self.size)
        label = loadLabel(self.label_path[index], self.size)
        return image, label
        
    def __len__(self):
        return len(self.image_path) 
   
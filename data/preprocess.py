from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import os


def loadColors():
    """ 
        从"colors.txt"文件中读取颜色索引对应的rgb颜色值。
    """
    colors = []
    with open("./data/colors.txt") as f:
        for line in f.readlines():
            # 去除空行
            if line.strip() == "":
                continue
            x = list(map(int, line.strip().split(',')))
            colors.append(x)
    colors = np.array(colors)
    return colors


def loadLabel(label_name, length, width):
    """
        读取标签图片，转化为固定大小，再将每个像素的颜色值转化为距离最近的颜色索引值
    Args:
        label_name: 标签图片文件名, *.jpg
        colors: 颜色索引
        label: 转化后的图片长宽

    Return:
        label: 转化后的标签图片 torch, shape: size * size
    """
    colors = loadColors()
    label = Image.open(label_name)
    transform = transforms.Compose([
                transforms.Scale((length, width))
    ])
    label = transform(label)
    label = np.array(label) 
    # label： size * size * 3

    label = np.expand_dims(label, axis=2)
    label = label.repeat(colors.shape[0], axis=2)
    diff = np.sum(np.square(label - colors), axis=3)
    label = np.argmin(diff, axis=2)
    label = (label == 1) * label
    label = torch.from_numpy(label)
    return label

def loadImage(image_name, length, width):
    """
        读取细胞原图片， 转化为固定大小
    Args:
        image_name: 细胞原图片文件名, *.jpg
        size: 转化后的图片长宽
    
    Return:
        image: 转化后的图片 torch, shape: 3 * size * size
    """
    image = Image.open(image_name)
    transform = transforms.Compose([
                transforms.Scale((length, width)),
                transforms.ToTensor()
    ])
    image = transform(image)
    return image


def checkImageLabel(id):
    """
        检验图片和标签的像素大小是否一致
    """
    image_name = "./data/image/image_" + str(id) + ".jpg"
    label_name = "./data/image/label_" + str(id) + ".jpg"
    image = np.array(Image.open(image_name))
    label = np.array(Image.open(label_name))
    print(image.shape)
    print(label.shape)    

def showImage(image):
    """
       将image以图片形式展示
    
    Args:
        label: 图片 torch, shape: 3 * size * size 
    """
    transform = transforms.ToPILImage()
    image = transform(image)
    plt.imshow(image)

def showLabel(label):
    """
        转化颜色索引， 将label以图片形式展示
    
    Args:
        label: 标签图片 torch, shape: size * size 
    """
    length = label.shape[0]
    width = label.shape[1]
    colors = loadColors()
    image = np.empty((length, width, 3), dtype=int)
    for i in range(length):
        for j in range(width):
            image[i][j] = colors[label[i][j]]
    plt.imshow(image)
    return image

def showSample(image, label, pred):
    """
        展示一组样例，包括图片, 标签, 预测标签

    Args:
        image: 原图片.   torch: 3 * size * size
        label: 标签图片. torch:  size * size
        pred: 预测图片.  torch: size * size 
    """
    image = torch.squeeze(image.cpu(), dim=0)
    label = torch.squeeze(label.cpu(), dim=0).numpy()
    pred = torch.squeeze(pred.cpu(), dim=0).numpy()
    plt.subplot(221)
    showImage(image)
   
    plt.subplot(223)
    showLabel(label)
    
    plt.subplot(224)
    showLabel(pred)
    
    plt.show()

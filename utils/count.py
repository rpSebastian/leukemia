import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from data.preprocess import showLabel

def read_image(id):
    img = cv2.imread(str(id) + ".jpg")
    return img


def remove_noise(img):
    # 对于opencv直接读取的图片格式为BGR, 一般图片格式为RGB
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
#     opening = cv2.erode(thresh,kernel,iterations=2)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    close = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 2)
#     sure_bg = cv2.dilate(opening,kernel,iterations=2)
    return close

def search_connected_block(sx, sy, img, mark, total):
    from collections import deque
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    n, m = img.shape
    q = deque()
    q.append((sx, sy))
    mark[sx][sy] = total
    while q:
        x, y = q.popleft()
        for k in range(4):
            xx, yy = x + dx[k], y + dy[k]
            if xx >= 0 and xx < n and yy >= 0 and yy < m and img[xx][yy] != 0 and not mark[xx][yy]:
                q.append((xx, yy))
                mark[xx][yy] = total

def count_region(img):
    mark = np.zeros_like(img)
    n, m = img.shape
    total = 0
    for i in range(n):
        for j in range(m):
            if img[i][j] != 0 and not mark[i][j]:
                total += 1
                search_connected_block(i, j, img, mark, total)
    return total, mark

def calc_distance_from_border(img):
    from collections import deque
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    n, m = img.shape
    q = deque()
    dis = np.zeros_like(img)
    vis = np.zeros_like(img)
    for i in range(n):
        for j in range(m):
            if img[i][j] == 0:
                vis[i][j] = 1
                q.append((i, j))
    while q:
        x, y = q.popleft()
        for k in range(4):
            xx, yy = x + dx[k], y + dy[k]
            if xx >= 0 and xx < n and yy >= 0 and yy < m and not vis[xx][yy]:
                q.append((xx, yy))
                vis[xx][yy] = 1
                dis[xx][yy] = dis[x][y] + 1
    return dis

def get_max_distance_per_region(img, mark, dis, total):
    max_dis = np.zeros((total + 1, ))
    n, m = img.shape
    for i in range(n):
        for j in range(m):
            if mark[i][j] != 0:
                max_dis[mark[i][j]] = max(max_dis[mark[i][j]], dis[i][j])
    return max_dis

def reduce_border_per_region(img, mark, dis, max_dis, reduce_rate):
    n, m = img.shape
    signal = False
    for i in range(n):
        for j in range(m):
            if img[i][j] != 0 and max_dis[mark[i][j]] > 2 and dis[i][j] <= 1:
#             if img[i][j] != 0 and dis[i][j] < int(max_dis[mark[i][j]] * reduce_rate):
                img[i][j] = 0
                signal = True
    return img, signal
def reduce_region(img, mark, total, reduce_rate):
    dis = calc_distance_from_border(img)
    max_dis = get_max_distance_per_region(img, mark, dis, total)
    img, signal = reduce_border_per_region(img, mark, dis, max_dis, reduce_rate)
    return img, signal

def eliminate_little_region(img, mark, total):
    for i in range(1, total + 1):
        if np.sum(mark==i) <= 2:
            img[mark==i] = 0
    return img

def count(img):
    print(img.shape)
    img = remove_noise(img)
    print(img.shape)
    showLabel(img)
    plt.show()
    signal = True
    while signal:
        total, mark = count_region(img)
        img, signal = reduce_region(img, mark, total, 0.8)
        img = eliminate_little_region(img, mark, total)
        showLabel(img)
        plt.show()
    return total


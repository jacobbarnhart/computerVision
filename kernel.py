from unittest import result
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from zmq import Errno

class kernel:
    def __init__(self, path):
        self.path = path
        self.img = color.rgb2gray(io.imread(self.path))
        self.img = np.array(self.img)


    def reset(self):
        self.img = color.rgb2gray(io.imread(self.path))
        self.img = np.array(self.img)
        
    def maxpool(self, size, stride):
        row, col = self.img.shape
        result = np.zeros((int(((row - size) / stride) + 1), int(((col - size) / stride) + 1)))
        for i in range(0, row - size, stride):
            for j in range(0, col - size, stride):
                max = np.max(self.img[i:i + size, j:j + size])
                result[int(((i - size) / stride) + 1), int(((j - size) / stride) + 1)] = max
        self.img = result


    def edge(self, size, stride, str):
        row, col = self.img.shape
        edgeKernel = self.edgeKernel(size, str)
        result = np.zeros((int(((row - size) / stride) + 1), int(((col - size) / stride) + 1)))
        for i in range(0, row - size, stride):
            for j in range(0, col - size, stride):
                result[int(((i - size) / stride) + 1), int(((j - size) / stride) + 1)] = np.sum(np.multiply(self.img[i:i + size, j:j + size], edgeKernel))
        self.img = result

    def sharpen(self, stride, str):
        row, col = self.img.shape
        size = 3
        sharpenKernel = np.array([[0, -1, 0], [-1, str, -1], [0, -1, 0]])
        result = np.zeros((int(((row - size) / stride) + 1), int(((col - size) / stride) + 1)))
        for i in range(0, row - size, stride):
            for j in range(0, col - size, stride):
                result[int(((i - size) / stride) + 1), int(((j - size) / stride) + 1)] = np.sum(np.multiply(self.img[i:i + size, j:j + size], sharpenKernel))
        self.img = result

    def edgeKernel(self, size, str):
        if size % 2 == 0:
            raise Exception("Size cannot be even!")
        result = np.ones((size, size))
        result = -1 * result
        result[int(size / 2), int(size / 2)] = str
        return result

    def plot(self):
        plt.imshow(self.img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        plt.show()

    # img = maxpool(img, 2, 1)


    # plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    # plt.show()

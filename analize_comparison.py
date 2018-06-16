import sys
import os
import collections
import struct
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

NNF1_PATH = os.path.expanduser('~/ann_l2.bmp')
NNF2_PATH = os.path.expanduser('~/ann_nn.bmp')
Pixel = collections.namedtuple('Pixel', ['x', 'y'])

class NNF:
    def __init__(self, image):
        self.image_ = image
        self.h_ = image.height
        self.w_ = image.width
        self.bytes_ = image.tobytes()


    def get_pixel(self, x, y):
        index = (y*self.w_ + x)*4
        b = self.bytes_[index : index+4]
        pixel_v = struct.unpack('<i', b)[0]
        # returning [x, y]
        return np.array([pixel_v & 0x0fff, pixel_v>>12 & 0x0fff])

    @property
    def height(self):
        return self.h_

    @property
    def width(self):
        return self.w_


def main():
    nnf1 = NNF(Image.open(NNF1_PATH))
    nnf2 = NNF(Image.open(NNF2_PATH))

    dist = np.zeros([nnf1.height, nnf1.width])

    for y in range(nnf1.height):
        for x in range(nnf1.width):
            nn1 = nnf1.get_pixel(x, y)
            nn2 = nnf2.get_pixel(x, y)
            dist[y][x] = np.linalg.norm(nn1 - nn2)

    imgplot = plt.imshow(dist)
    imgplot.set_cmap('jet')
    plt.colorbar()
    plt.title('Nearest neighbor difference between \noriginal PatchMatch and Patch2Vec variant')

    plt.savefig('ann_algorithms_distance.png')

if __name__ == '__main__':
    sys.exit(main())

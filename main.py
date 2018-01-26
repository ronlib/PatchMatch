import sys
import argparse
import matplotlib
import matplotlib.image as mpimg
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import config

def main():

    parser = argparse.ArgumentParser(description='Image completion example')
    parser.add_argument('-f', '--file', required=True, dest='input_file_path', help='input file path')

    args = parser.parse_args()

    figure_paint = FigurePaint(args.input_file_path)
    plt.show()
    while True:
        plt.waitforbuttonpress()


class FigurePaint(object):

    def __init__(self, image_path):
        self.fig, self.ax = plt.subplots(1,1)
        self.fig_mask, self.ax_mask = plt.subplots(1,1)
        self.img = mpimg.imread(image_path)
        self.ax.imshow(self.img)
        self.image_mask = np.ndarray(self.img.shape[0:2], dtype=np.int8)
        self.image_mask[:,:] = 0
        self.fig.suptitle('Image')
        self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_event)

    def mouse_event(self, event):
        if not event.inaxes:
            return

        # ipdb.set_trace()
        if event.button == 1:
            x, y = int(event.xdata), int(event.ydata)
            # self.image_mask.itemset((y, x),1)
            x_min, y_min = max(0, x-config.DRAW_PATCH_SIZE//2), max(0, y-config.DRAW_PATCH_SIZE//2)
            x_max, y_max = min(self.img.shape[1], x+config.DRAW_PATCH_SIZE//2), min(self.img.shape[0], y+config.DRAW_PATCH_SIZE//2)

            print (x_min, x_max, y_min, y_max)
            self.ax.add_patch(patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, color='w'))
            self.image_mask[y_min:y_max, x_min:x_max] = 1
            self.ax_mask.imshow(self.image_mask, cmap='binary')
            self.fig.canvas.draw()
            self.fig_mask.canvas.draw()
            print (event.xdata, event.ydata)





if __name__ == '__main__':
    main()

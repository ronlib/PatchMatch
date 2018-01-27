import sys
import argparse
import matplotlib
import matplotlib.image as mpimg
from matplotlib.widgets import Button
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import config
import pyinpaint

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
        self.image_mask = np.ndarray(self.img.shape[:2], dtype=np.uint8)
        self.image_mask[:,:] = 0
        self.fig.suptitle('Image')
        self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_event)
        self.ax_inpaint_button = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        self.button_inpaint = Button(self.ax_inpaint_button, 'inpaint')
        self.fig.canvas.mpl_connect('button_press_event', self.inpaint_click)
        # self.button_inpaint.on_clicked(inpaint_click)

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
            self.image_mask[y_min:y_max, x_min:x_max] = 255
            self.ax_mask.imshow(self.image_mask, cmap='binary')
            self.fig.canvas.draw()
            self.fig_mask.canvas.draw()
            print (event.xdata, event.ydata)

    def inpaint_click(self, event):
        (xmin, ymin), (xmax, ymax) = self.button_inpaint.label.clipbox.get_points()
        if xmin<event.x<xmax and ymin<event.y<ymax:
            print ("Inpainting! (not yet)")
            print ("Image dimensions: ", self.img.shape)
            print ("Mask dimensions: ", self.image_mask.shape)
            print ("Mask size: ", self.image_mask.size)
            inp_img_buf = pyinpaint.pyinpaint(self.img, 3, self.image_mask, self.img.shape[0], self.img.shape[1])
            print (inp_img_buf.shape)
            inp_img_buf = inp_img_buf.reshape(self.img.shape)
            self.ax.clear()
            self.ax.imshow(inp_img_buf)
            self.fig.canvas.draw()



if __name__ == '__main__':
    main()

from skimage import io
import matplotlib.pyplot as pl
import numpy as np
from bubble_tools import split_image

test_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/Image73.lsm'
seg_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/Classified_image73.tif'

im = io.imread(test_file)[0]
pl.imshow(im)
pl.show()

seg_im = io.imread(seg_file, as_gray=True)[0]
pl.imshow(seg_im)
pl.show()

seg_im[0, 0]  # a green value
seg_im[300, 300]  # a red value

type(im)

new_im = np.where(seg_im < 0.5, -100, im)
pl.imshow(new_im)
pl.show()

size_x = new_im.shape[1]
size_y = new_im.shape[0]
chunk_x = 128
chunk_y = 128
shift = 64

n_y_shift = (size_y-chunk_y)//shift
n_x_shift = (size_x-chunk_x)//shift
n_y_shift
n_x_shift

x_min = 0
x_max = chunk_x
for i in range(n_y_shift):
    y_min = 0
    y_max = chunk_y
    for j in range(n_x_shift):
        im_tile = new_im[y_min:y_max, x_min:x_max]
        if -100 not in im_tile:
            pl.imshow(im_tile)
            pl.title(str(i) + ", " + str(j))
            pl.show()
        y_min += shift
        y_max += shift
    x_min += shift
    x_max += shift


# %% testing out the function
im_list, cent_list, d_list = split_image(im, seg_im, 128, 128, 64, plot_ims=True)

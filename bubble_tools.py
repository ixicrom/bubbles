import numpy as np
import matplotlib.pyplot as pl
from skimage import io

# %% test variables
# test_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/Image73.lsm'
# seg_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/Classified_image73.tif'
# im = io.imread(test_file)[0]
# seg_im = io.imread(seg_file, as_gray=True)[0]
# plot_ims = False
# chunk_x = 128
# chunk_y = 128
# shift = 16


# %%
def find_closest(im_centre, hole_locs):
    dists = []
    for px in hole_locs:
        dy = im_centre[0] - px[0]
        dx = im_centre[1] - px[1]
        dist = np.sqrt(dx**2 + dy**2)
        dists.append(int(dist))
    return min(dists)


def split_image(im, seg_im, chunk_x, chunk_y, shift, plot_ims=False):
    im_list = []
    centre_list = []
    dist_list = []
    new_im = np.where(seg_im < 0.5, -100, im)

    holes = np.where(new_im == -100)
    hole_locs = list(zip(holes[0], holes[1]))

    size_x = new_im.shape[1]
    size_y = new_im.shape[0]

    n_y_shift = (size_y-chunk_y)//shift
    n_x_shift = (size_x-chunk_x)//shift

    x_min = 0
    x_max = chunk_x
    for i in range(n_y_shift):
        y_min = 0
        y_max = chunk_y
        for j in range(n_x_shift):
            im_tile = new_im[y_min:y_max, x_min:x_max]

            if -100 not in im_tile:
                im_list.append(im_tile)
                centre = (y_min+chunk_y//2, x_min+chunk_x//2)
                centre_list.append(centre)
                dist = find_closest(centre, hole_locs)
                dist_list.append(dist)
                if plot_ims:
                    pl.imshow(im_tile)
                    pl.title(str(centre)+", "+str(dist)+" pixels from hole")
                    pl.show()

            y_min += shift
            y_max += shift
        x_min += shift
        x_max += shift
    return im_list, centre_list, dist_list

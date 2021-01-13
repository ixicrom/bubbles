import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from scipy import fft
from scipy import ndimage
from skimage import io

# %% test variables
# test_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/Image73.lsm'
# seg_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/model_73/Classified_image73.tif'
# im = io.imread(test_file)[0]
# seg_im = io.imread(seg_file)[0]
# plot_ims = False
# chunk_x = 128
# chunk_y = 128
# shift = 16


# %% functions to split image into chunks
def find_closest(im_centre, hole_locs):
    dists = []
    for px in hole_locs:
        dy = im_centre[0] - px[0]
        dx = im_centre[1] - px[1]
        dist = np.sqrt(dx**2 + dy**2)
        dists.append(int(dist))
    return min(dists)


def split_image(im, seg_im, bijel_val, chunk_x, chunk_y, shift, plot_ims=False):
    im_list = []
    centre_list = []
    dist_list = []

    # set real image to -100 where the segmented image is not a bijel
    new_im = np.where(seg_im != bijel_val, -100, im)

    # get locations of holes
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
        dat = pd.DataFrame([im_list, centre_list, dist_list])
        dat = dat.transpose()
        dat.columns = ['chunk_im', 'chunk_loc', 'distance']
    print(dat.shape)
    return dat


# %% functions to calculate ACF
def x_profile(im):
    x_profile = []
    y, x = np.indices(im.shape)
    for i in np.unique(x):
        x_profile.append(np.mean(im[x == i]))
    return x_profile


def y_profile(im):
    y_profile = []
    y, x = np.indices(im.shape)
    for i in np.unique(y):
        y_profile.append(np.mean(im[y == i]))
    return y_profile


def xy_autocorr(im):
    ft = fft.fft2(im)
    ft_conj = np.conj(ft)
    m, n = ft.shape
    acf = np.real(fft.ifft2(ft*ft_conj))
    acf = np.roll(acf, -m//2+1, axis=0)
    acf = np.roll(acf, -n//2+1, axis=1)
    return acf


# %% functions to extract variables from ACF
def turnpoints(lst):
    x = np.array(lst)
    n = len(x)
    x0 = x[0]-1.
    x2 = np.concatenate(([x0], x[:-1]))
    diffs = x != x2
    uniques = x[diffs]
    uniques
    n2 = len(uniques)
    poss = np.arange(n)[diffs]
    exaequos = np.concatenate((poss[1:n2], [n+1]))-poss-1
    '''
    at some point need to add in if statements to catch when
    things are wrong as with the R package
    '''
    m = n2-2
    vals = np.concatenate((np.arange(m)+2, np.arange(m)+1, np.arange(m)))
    ex = np.array(uniques[vals])
    ex = np.reshape(ex, (-1, m))
    ex = np.transpose(ex)
    peaks = [False]
    pits = [False]
    for i in range(m):
        peaks.append(ex[i, 1] == max(ex[i, ]))
        pits.append(ex[i, 1] == min(ex[i, ]))
    peaks.append(False)
    pits.append(False)
    tpts = [a or b for a, b in zip(peaks, pits)]
    if sum(tpts) == 0:
        tppos = np.nan
        peaks = [False]*n2
        pits = [False]*n2
    else:
        tppos = (poss+exaequos)[tpts]
    return tppos


def acf_variables(acf_x, acf_y):
    x_size = len(acf_x)
    y_size = len(acf_y)

    acf_x = acf_x[x_size//2+1:]
    acf_y = acf_y[y_size//2+1:]

    grad_x = (acf_x[4] - acf_x[0])/5
    grad_y = (acf_y[4] - acf_y[0])/5
    grad_diff = grad_x - grad_y

    try:
        tp_x = turnpoints(acf_x)[0]
        tp_y = turnpoints(acf_y)[0]
        tp_diff = tp_x - tp_y
    except TypeError:
        tp_x = None
        tp_y = None
        tp_diff = None

    output = {'grad_x': grad_x,
              'grad_y': grad_y,
              'grad_diff': grad_diff,
              'tp_x': tp_x,
              'tp_y': tp_y,
              'tp_diff': tp_diff}
    return pd.Series(output)


# %% sobel filter for looking at anisotropy
def sobel(im):
    x_kernel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    y_kernel = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 2]])
    kernel_45 = np.array([[-1, -1, 2],
                          [-1, 2, -1],
                          [2, -1, -1]])
    kernel_neg45 = np.array([[2, -1, -1],
                             [-1, 2, -1],
                             [-1, -1, 2]])
    sobel_x = ndimage.convolve(im, x_kernel)
    sobel_y = ndimage.convolve(im, y_kernel)
    sobel_45 = ndimage.convolve(im, kernel_45)
    sobel_neg45 = ndimage.convolve(im, kernel_neg45)

    return sobel_x, sobel_y, sobel_45, sobel_neg45


# %% functions for calculating orientation of chunks and segmented images
def seg_orientation(seg_im_array, hole_pixel_bool):
    # pl.imshow(seg_im_array)
    # pl.colorbar()
    # pl.show()
    im_x, im_y = np.meshgrid(np.arange(seg_im_array.shape[1]),
                             np.arange(seg_im_array.shape[0]))
    im_table = np.vstack((im_x.ravel(), im_y.ravel(), seg_im_array.ravel())).T
    im_table
    im_df = pd.DataFrame(im_table, columns=['x', 'y', 'val'])
    im_df['val'].unique()

    pixels = im_df[hole_pixel_bool][['x', 'y']].values.T

    if pixels.shape[1] > 1:
        cov_matrix = np.cov(pixels)
        eigen = np.linalg.eig(cov_matrix)
        e_vals = eigen[0]
        e_vecs = eigen[1]
        e_vec = e_vecs[np.argmax(e_vals)]
        angle = np.arctan(e_vec[0]/e_vec[1])
    angle
    return angle


def orientation(im_array, n_bins=10, plot=False):
    im_max = np.max(im_array)
    im_min = np.min(im_array)
    bins = np.linspace(im_min, im_max, n_bins)
    im_bins = np.digitize(im_array, bins)
    if plot:
        pl.contour(im_array)
        pl.show()
        pl.imshow(im_bins)
        pl.show()
    im_x, im_y = np.meshgrid(np.arange(im_bins.shape[1]),
                             np.arange(im_bins.shape[0]))
    im_table = np.vstack((im_x.ravel(), im_y.ravel(), im_bins.ravel())).T
    im_table
    im_df = pd.DataFrame(im_table, columns=['x', 'y', 'val'])
    val_list = []
    e_val_list = []
    e_vec_list = []
    angle_list = []
    e_ratio_list = []
    for val in im_df['val'].unique():
        pixels = im_df[im_df['val'] == val][['x', 'y']].values.T

        if pixels.shape[1] > 1:
            cov_matrix = np.cov(pixels)
            eigen = np.linalg.eig(cov_matrix)
            e_vals = eigen[0]
            e_vecs = eigen[1]
            # print('max')
            # print(np.argmax(e_vals))
            e_val = np.max(e_vals)
            # print(e_val)
            e_vec = e_vecs[np.argmax(e_vals)]
            e_val_2 = e_vals[np.argsort(e_vals)[0]]
            # print(e_val_2)
            # print('second')
            # print(np.argsort(e_vals)[0])
            angle = np.arctan(e_vec[0]/e_vec[1])
            val_list.append(val)
            e_vec_list.append(e_vec)
            e_val_list.append(e_val)
            e_ratio_list.append(e_val/e_val_2)
            angle_list.append(angle)

    out = pd.DataFrame([val_list,
                        e_vec_list, e_val_list,
                        e_ratio_list, angle_list]).T
    out.columns = ['val', 'e_vec', 'e_val', 'e_ratio', 'angle']
    return out

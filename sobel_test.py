from bubble_tools import split_image, x_profile, y_profile, xy_autocorr, acf_variables, sobel
from skimage import io
import os
import pandas as pd
import matplotlib.pyplot as pl

dat_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/dat_list_73.csv'
im_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/'
seg_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/model_73'

# %% reading files and opening images


dat_liquid = pd.DataFrame({'chunk_im': [],
                            'chunk_loc': [],
                            'distance': []})
dat_particle = pd.DataFrame({'chunk_im': [],
                             'chunk_loc': [],
                             'distance': []})

f = open(dat_file, 'r')
for line in f.readlines():
    if not line.endswith('x') and line.startswith('Image'):
        vals = line.split(',')

        im_file = os.path.join(im_folder, vals[0])
        liquid_im = io.imread(im_file)[0]
        particle_im = io.imread(im_file)[1]

        seg_file = os.path.join(seg_folder, vals[1])
        seg_im = io.imread(seg_file, as_gray=True)[0]

        liquid = split_image(liquid_im, seg_im, 128, 128, 64)
        dat_liquid = dat_liquid.append(liquid)

        particle = split_image(particle_im, seg_im, 128, 128, 64)
        dat_particle = dat_particle.append(particle)

f.close()

dat_liquid = dat_liquid.reset_index(drop=True)
dat_liquid = dat_liquid.astype({'chunk_im': 'object',
                                  'chunk_loc': 'object',
                                  'distance': 'float64'})
print(dat_liquid.describe())

dat_particle = dat_particle.reset_index(drop=True)
dat_particle = dat_particle.astype({'chunk_im': 'object',
                                  'chunk_loc': 'object',
                                  'distance': 'float64'})
print(dat_particle.describe())


# %% sobel filters
# test_im = dat_overlap_all['chunk_im'][0]


# def sobel(im):
#     x_kernel = np.array([[-1, 0, 1],
#                          [-2, 0, 2],
#                          [-1, 0, 1]])
#     y_kernel = np.array([[-1, -2, -1],
#                          [0, 0, 0],
#                          [1, 2, 2]])
#     kernel_45 = np.array([[-1, -1, 2],
#                           [-1, 2, -1],
#                           [2, -1, -1]])
#     kernel_neg45 = np.array([[2, -1, -1],
#                              [-1, 2, -1],
#                              [-1, -1, 2]])
#     sobel_x = ndimage.convolve(im, x_kernel)
#     sobel_y = ndimage.convolve(im, y_kernel)
#     sobel_45 = ndimage.convolve(im, kernel_45)
#     sobel_neg45 = ndimage.convolve(im, kernel_neg45)
#
#     return sobel_x, sobel_y, sobel_45, sobel_neg45


x_ims = []
y_ims = []
p45_ims = []
n45_ims = []
for im in dat_particle['chunk_im']:
    sobel_x, sobel_y, sobel_45, sobel_neg45 = sobel(im)

    x_ims.append(sobel_x)
    y_ims.append(sobel_y)
    p45_ims.append(sobel_45)
    n45_ims.append(sobel_neg45)

    # pl.subplot(2, 2, 1)
    # pl.imshow(sobel_x)
    # pl.title('Sobel filter x')
    #
    # pl.subplot(2, 2, 2)
    # pl.imshow(sobel_y)
    # pl.title('Sobel filter y')
    #
    # pl.subplot(2, 2, 3)
    # pl.imshow(sobel_45)
    # pl.title('Sobel filter +45deg')
    #
    # pl.subplot(2, 2, 4)
    # pl.imshow(sobel_neg45)
    # pl.title('Sobel filter -45deg')
    # pl.tight_layout()
    # pl.show()

dat_particle['x'] = x_ims
dat_particle['y'] = y_ims
dat_particle['45deg'] = p45_ims
dat_particle['-45deg'] = n45_ims

# %%
dat_particle.head()

from bubble_tools import xy_autocorr, split_image
from skimage import io
import os
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import imageio

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

        seg_file = os.path.join(seg_folder, vals[1])
        seg_im = io.imread(seg_file, as_gray=True)[0]

        liquid = split_image(liquid_im, seg_im, 128, 128, 64)
        dat_liquid = dat_liquid.append(liquid)


f.close()

dat_liquid = dat_liquid.reset_index(drop=True)
dat_liquid = dat_liquid.astype({'chunk_im': 'object',
                                'chunk_loc': 'object',
                                'distance': 'float64'})
print(dat_liquid.head())



# %% calculate ACF for each image (just looking at liquid channel)
dat_acf = []
i = 0
for im in dat_liquid['chunk_im']:
    acf = xy_autocorr(im)
    dat_acf.append(acf)
    imageio.imwrite(str(i)+'test.png', acf)
    i += 1
    # cov_matrix = np.cov(acf)
    # cov_matrix.shape
    # eigen = np.linalg.eig(cov_matrix)
    # e_vals = eigen[0]
    # e_vecs = eigen[1]
    # np.argmax(e_vals)
    # np.max(e_vals)
    # e_vecs[np.argmax(e_vals)]


dat_liquid['acf'] = dat_acf

print(dat_liquid.head())

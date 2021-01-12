from bubble_tools import xy_autocorr, split_image, seg_orientation
from skimage import io
import os
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, style
# style.core.reload_library()
# pl.style.use('thesis')

dat_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/dat_list_region-grow.csv'
im_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/'
seg_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/region_growing/'

dat = pd.DataFrame({'chunk_im': [],
                    'chunk_loc': [],
                    'distance': []})

seg_angles = []

f = open(dat_file, 'r')
for line in f.readlines():
    if line.startswith('Image') and not line.endswith('x\n'):
        vals = line.split(',')

        im_file = os.path.join(im_folder, vals[0])
        im = io.imread(im_file)[0]
        # io.imshow(im)
        # pl.show()

        seg_file = os.path.join(seg_folder, vals[1])
        seg_im = io.imread(seg_file)
        # io.imshow(seg_im)
        # pl.show()

        # chunks = split_image(im=im, seg_im=seg_im, bijel_val=2,
                             # chunk_x=64, chunk_y=64, shift=32)
        # dat = dat.append(chunks)
        # seg_angle = seg_orientation(seg_im)
        # for i in range(chunks.shape[0]):
            # seg_angles.append(seg_angle)



seg_im_array = seg_im
pl.imshow(seg_im_array)
im_x, im_y = np.meshgrid(np.arange(seg_im_array.shape[1]),
                         np.arange(seg_im_array.shape[0]))
im_table = np.vstack((im_x.ravel(), im_y.ravel(), seg_im_array.ravel())).T

im_table

im_df = pd.DataFrame(im_table, columns=['x', 'y', 'val'])
im_df['val'].unique()
pixels = im_df[im_df['val'] != 2]


# if pixels.shape[1] > 1:
#     cov_matrix = np.cov(pixels)
#     eigen = np.linalg.eig(cov_matrix)
#     e_vals = eigen[0]
#     e_vecs = eigen[1]
#     e_vec = e_vecs[np.argmax(e_vals)]
#     angle = np.arctan(e_vec[0]/e_vec[1])
# angle
print('starting cov calculation...')
test=np.cov(pixels)
print('calculation complete')
print(test)

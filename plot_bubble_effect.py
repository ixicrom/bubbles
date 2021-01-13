from skimage import io
import os
import matplotlib.pyplot as pl
import numpy as np
from bubble_tools import find_closest
import cv2

while True:
    seg_type = input('Which segmentation data? 0=weka, 1=region_growing: ')
    if seg_type == '1':
        seg_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/region_growing/'
        print('Using segmented images from weka folder')
        break
    elif seg_type == '0':
        seg_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/weka_best/'
        print('Using segmented images from region growing folder')
        break
    else:
        print('Invalid selection, try again')
im_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/'
dat_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/dat_list.csv'

boundary_thickness = 150

f = open(dat_file, 'r')
for line in f.readlines():
    if line.startswith('Image') and not line.endswith('x\n'):
        vals = line.split(',')

        im_file = os.path.join(im_folder, vals[0])
        im = io.imread(im_file)[0]
        # io.imshow(im)
        # pl.show()

        seg_file = os.path.join(seg_folder, vals[1])
        if seg_type == '1':
            seg_im = io.imread(seg_file)
            hole_bool = (seg_im != 2)
            b_val = 2

        else:
            seg_im = io.imread(seg_file)[0]
            hole_bool = (seg_im == 0)
            b_val = 1


        hole_im = np.where(hole_bool, 1, 0).reshape(512, 512)
        edges, blarg = cv2.findContours((hole_im*255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        xx, yy  = np.meshgrid(np.arange(512), np.arange(512))
        boundary_mask = np.zeros((512,512))
        for edge in edges:
            for point in edge:
                point = point[0]
                xc = point[0]
                yc = point[1]
                circle = (xx-xc)**2 + (yy-yc)**2 < boundary_thickness**2
                boundary_mask[circle] = 1
        seg_plot = np.where(hole_bool, 0, 2).reshape(512, 512)
        boundary_area = np.where(hole_bool, hole_im*2, boundary_mask)
        # pl.imshow(boundary_area)

        seg_im.shape
        pl.imshow(im, cmap='gray')
        pl.imshow(boundary_area, alpha=0.5, vmin=0, vmax=2)
        pl.show()
f.close()

from bubble_tools import xy_autocorr, split_image, seg_orientation, orientation
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

while True:
    seg_type = input('Which segmentation data? 0=weka, 1=region_growing: ')
    if seg_type == '1':
        dat_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/dat_list_region-grow.csv'
        seg_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/region_growing/'
        print('Using segmented images from weka folder')
        break
    elif seg_type == '0':
        dat_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/dat_list_weka.csv'
        seg_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/weka_best/'
        print('Using segmented images from region growing folder')
        break
    else:
        print('Invalid selection, try again')
im_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/'

while True:
    or_tag = input('Override image list with dat_list.csv? y/n: ')
    if or_tag == 'y':
        dat_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/dat_list.csv'
        print('Using ' + dat_file)
        break
    elif or_tag == 'n':
        print('Using ' + dat_file)
        break
    else:
        print('Invalid selection, try again')

dat = pd.DataFrame({'chunk_im': [],
                    'chunk_loc': [],
                    'distance': []})

while True:
    plot_tag = input('Plot segmented images with angle? y/n: ')
    if plot_tag == 'y':
        plot_bool = True
        break
    elif plot_tag == 'n':
        plot_bool = False
        break
    else:
        print('Invalid selection, try again')

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
        if seg_type == '1':
            seg_im = io.imread(seg_file)
            hole_bool = (seg_im != 2).ravel()
            b_val = 2

        else:
            seg_im = io.imread(seg_file)[0]
            hole_bool = (seg_im == 0).ravel()
            b_val = 1
        # io.imshow(seg_im)
        # pl.show()

        chunks = split_image(im=im, seg_im=seg_im, bijel_val=b_val,
                             chunk_x=64, chunk_y=64, shift=32)
        dat = dat.append(chunks)

        seg_angle = seg_orientation(seg_im, hole_bool)
        for i in range(chunks.shape[0]):
            seg_angles.append(seg_angle)
        print(math.degrees(seg_angle))

        if plot_bool:
            io.imshow(seg_im)
            pl.title('Angle (degrees) = '+str(math.degrees(seg_angle)))
            pl.show()

f.close()

dat = dat.reset_index(drop=True)
dat = dat.astype({'chunk_im': 'object',
                  'chunk_loc': 'object',
                  'distance': 'float64'})
dat['seg_angle'] = seg_angles

dat_acf = []
av_orientation = []
e_ratio = []
for im in dat['chunk_im']:
    acf = xy_autocorr(im)
    dat_acf.append(acf)
    orient = orientation(acf)
    av_orientation.append(np.mean(orient['angle']))
    e_ratio.append(np.mean(orient['e_ratio']))

dat['av_or'] = av_orientation
dat['e_ratio'] = e_ratio

dat.head()

dat_plot = dat.sort_values('e_ratio').replace([np.inf, -np.inf],
                                              np.nan).dropna()

dat_plot['angle_deg'] = (dat_plot['av_or']-dat_plot['seg_angle'])*180/math.pi

pl.scatter(dat_plot['distance'],
           dat_plot['angle_deg'],
           c=dat_plot['e_ratio'])
pl.xlabel('Distance from bubble trace')
pl.ylabel('Angle (degrees)')
# pl.title('Region growing segmentation')
pl.colorbar()
pl.show()

# %% try a 3D bar plot

fig = pl.figure()
ax = Axes3D(fig)

x = dat_plot['distance'].values
y = (dat_plot['angle_deg'].values)
np.max(x)
np.min(x)
np.min(y)
np.max(y)
hist, xedges, yedges = np.histogram2d(x, y, bins=(40, 20))

xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:],
                         yedges[:-1]+yedges[1:],
                         indexing="ij")
xpos = xpos.ravel()*0.5
ypos = ypos.ravel()*0.5
zpos = np.zeros_like(xpos)

dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]
dz = hist.ravel()

cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
max_height = np.max(dz)   # get range of colorbars so we can normalize
min_height = np.min(dz)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in dz]

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
pl.xlabel('Distance')
pl.ylabel('Angle (degrees)')
pl.title('Region growing segmentation')
pl.show()

# %%

dist_0deg = dat_plot[(dat_plot['angle_deg'] < 25) & (dat_plot['angle_deg'] > -25)]['distance']
dist_0deg.hist()
pl.xlabel('Distance')
pl.ylabel('Frequency')
pl.title('Angle = 0 degrees (+/- 25deg)')
pl.show()

dist_75deg = dat_plot[(dat_plot['angle_deg'] < 75) & (dat_plot['angle_deg'] > 25)]['distance']
dist_75deg.hist()
pl.xlabel('Distance')
pl.ylabel('Frequency')
pl.title('Angle = 50 degrees (+/- 25deg)')
pl.show()

dist_n75deg = dat_plot[(dat_plot['angle_deg'] > -75) & (dat_plot['angle_deg'] < -25)]['distance']
dist_n75deg.hist()
pl.xlabel('Distance')
pl.ylabel('Frequency')
pl.title('Angle = -50 degrees (+/- 25deg)')
pl.show()

ang_100px = dat_plot[(dat_plot['distance'] > 75) & (dat_plot['distance'] < 125)]['angle_deg']
ang_100px.hist()
pl.xlabel('Angle (deg)')
pl.ylabel('Frequency')
pl.title('Distance = 100 pixels (+/- 25px)')
pl.show()


ang_200px = dat_plot[(dat_plot['distance'] > 175) & (dat_plot['distance'] < 225)]['angle_deg']
ang_200px.hist()
pl.xlabel('Angle (deg)')
pl.ylabel('Frequency')
pl.title('Distance = 200 pixels (+/- 25px)')
pl.show()

ang_300px = dat_plot[(dat_plot['distance'] > 275) & (dat_plot['distance'] < 325)]['angle_deg']
ang_300px.hist()
pl.xlabel('Angle (deg)')
pl.ylabel('Frequency')
pl.title('Distance = 300 pixels (+/- 25px)')
pl.show()


ang_100px_abs = np.abs(dat_plot[(dat_plot['distance'] > 75) & (dat_plot['distance'] < 125)]['angle_deg'])
ang_100px_abs.hist()
pl.xlabel('Magnitude of angle (deg)')
pl.ylabel('Frequency')
pl.title('Distance = 100 pixels (+/- 25px)')
pl.show()

ang_200px_abs = np.abs(dat_plot[(dat_plot['distance'] > 175) & (dat_plot['distance'] < 225)]['angle_deg'])
ang_200px_abs.hist()
pl.xlabel('Magnitude of angle (deg)')
pl.ylabel('Frequency')
pl.title('Distance = 200 pixels (+/- 25px)')
pl.show()

ang_300px_abs = np.abs(dat_plot[(dat_plot['distance'] > 275) & (dat_plot['distance'] < 325)]['angle_deg'])
ang_300px_abs.hist()
pl.xlabel('Magnitude of angle (deg)')
pl.ylabel('Frequency')
pl.title('Distance = 300 pixels (+/- 25px)')
pl.show()


# %% try plotting average angle as function of distance

xedges
angle_bin_means = []
for i in range(len(xedges)):
    if i != 0:
        xmin = xedges[i-1]
        xmax = xedges[i]
        # print(str(xmin)+', '+str(xmax))
        # print(xmax-xmin)
        dat_bins = dat_plot[(dat_plot['distance'] > xmin) & (dat_plot['distance'] < xmax)]
        angle_bins = dat_bins['angle_deg']
        angle_bin_means.append(angle_bins.mean())

dat_bins
angle_bins

angle_bin_means
pl.scatter(xedges[:-1], angle_bin_means)
pl.xlabel('Distance from bubble trace (lower limit of bin)')
pl.ylabel('Average orientation of autocorrelation function')
pl.show()

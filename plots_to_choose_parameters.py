from bubble_tools import seg_orientation, split_image, orientation, xy_autocorr
import pandas as pd
import os
from skimage import io
import math
import matplotlib.pyplot as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, style

# initialise file suffix and folder for saving
file_suffix = ''
save_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/plots/choose_parameters/'

# define files for segmented images
while True:
    seg_type = input('Which segmentation data? weka or rg: ')
    if seg_type == 'rg':
        seg_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/region_growing/'
        print('Using segmented images from region growing folder')
        break
    elif seg_type == 'weka':
        seg_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/weka_best/'
        print('Using segmented images from weka folder')
        break
    else:
        print('Invalid selection, try again')
file_suffix += '_' + seg_type

# define image list file and folder to find images
dat_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/dat_list.csv'
im_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/'

# decide whether to plot segmented images with their orientation angle
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

# decide size of chunks to split data into
chunk_size = int(input('What size chunks (in pixels)?: '))
file_suffix += '_' + str(chunk_size)

# initialise empty dataframe and lists
dat = pd.DataFrame({'chunk_im': [],
                    'chunk_loc': [],
                    'distance': []})
seg_angles = []
sample_ids = []
# read through the image list file
f = open(dat_file, 'r')
for line in f.readlines():
    # check it's an image I want
    if line.startswith('Image') and not line.endswith('x\n'):
        vals = line.split(',')

        # read in the data image
        im_file = os.path.join(im_folder, vals[0])
        im = io.imread(im_file)[0]

        # read in the segmented images
        seg_file = os.path.join(seg_folder, vals[1])
        if seg_type == 'rg':
            seg_im = io.imread(seg_file)
            hole_bool = (seg_im != 2).ravel()
            b_val = 2

        else:
            seg_im = io.imread(seg_file)[0]
            hole_bool = (seg_im == 0).ravel()
            b_val = 1

        # split the image into chunks using info from the segmented image
        # uses shift = chunk size so no overlap
        chunks = split_image(im=im, seg_im=seg_im, bijel_val=b_val,
                             chunk_x=64, chunk_y=64,
                             shift=32)
        dat = dat.append(chunks)

        # calulate the orientation of the segmented image and store
        seg_angle = seg_orientation(seg_im, hole_bool)
        sample_id = vals[2]
        for i in range(chunks.shape[0]):
            seg_angles.append(seg_angle)
            sample_ids.append(sample_id)
        print(math.degrees(seg_angle))

        # plot the segmented images with their orientation if wanted
        if plot_bool:
            io.imshow(seg_im)
            pl.title('Angle (degrees) = '+str(math.degrees(seg_angle)))
            im_save = vals[0] + file_suffix + '.png'
            pl.savefig(os.path.join(save_folder, im_save))
            pl.show()

f.close()

# neaten up dataframe
dat = dat.reset_index(drop=True)
dat = dat.astype({'chunk_im': 'object',
                  'chunk_loc': 'object',
                  'distance': 'float64'})

# add calculated hole angles to dataframe
dat['sample_id'] = sample_ids
dat['seg_angle'] = seg_angles

# calculate orientation of each chunk and add to dataframe
av_orientation = []
for im in dat['chunk_im']:
    acf = xy_autocorr(im)
    orient = orientation(acf)
    av_orientation.append(np.mean(orient['angle']))
dat['av_or'] = av_orientation
dat.head()

# calculate the angle relevant to the hole orientation and turn to degrees
dat['angle_deg'] = (dat['av_or']-dat['seg_angle'])*180/math.pi

# make 3D histogram
fig = pl.figure()
ax = Axes3D(fig)

x = dat['distance'].values
y = (dat['angle_deg'].values)
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
file_3d = 'hist3d' + file_suffix + '.png'
pl.savefig(os.path.join(save_folder, file_3d))
pl.show()

# plot histogram of distances around 0 degrees of angle
dist_0deg = dat[(dat['angle_deg'] < 25) & (dat['angle_deg'] > -25)]['distance']
dist_0deg.hist()
pl.xlabel('Distance')
pl.ylabel('Frequency')
pl.title('Angle = 0 degrees (+/- 25deg)')
file_0deg = 'hist_0deg' + file_suffix + '.png'
pl.savefig(os.path.join(save_folder, file_0deg))
pl.show()

# try plot average angle as function of distance
# uses xedges as calculated for the 3D histogram
angle_bin_means = []
for i in range(len(xedges)):
    if i != 0:
        xmin = xedges[i-1]
        xmax = xedges[i]
        # print(str(xmin)+', '+str(xmax))
        # print(xmax-xmin)
        dat_bins = dat[(dat['distance'] > xmin) & (dat['distance'] < xmax)]
        angle_bins = dat_bins['angle_deg']
        angle_bin_means.append(angle_bins.mean())

dat_bins
angle_bins

angle_bin_means
pl.scatter(xedges[:-1], angle_bin_means)
pl.xlabel('Distance from bubble trace (lower limit of bin)')
pl.ylabel('Average orientation of autocorrelation function')
file_ang_dist = 'ang-dist' + file_suffix + '.png'
pl.savefig(os.path.join(save_folder, file_ang_dist))
pl.show()

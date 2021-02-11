import matplotlib.pyplot as pl
import os
from skimage import io
import math
from bubble_tools import seg_orientation, xy_autocorr, orientation, split_image, get_angle
import numpy as np
import pandas as pd


# define function for plotting the angle on an image
def abline(angle_rad, intercept):
    slope = math.tan(angle_rad)
    axes = pl.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    pl.plot(x_vals, y_vals, '--', color='r')


file_suffix = ''
save_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/plots/rg_tuning/'
seg_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/region_growing/'
dat_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/dat_list_region-grow.csv'
im_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/'

# decide size of chunks to split data into
chunk_size = int(input('What size chunks (in pixels)?: '))
file_suffix += '_' + str(chunk_size)

# decide which type of angle to use for segmented images
acf_angle = input('Use angle from ACF of segmented image instead of raw? y/n: ') =='y'

# initialise empty dataframe and lists
dat = pd.DataFrame({'chunk_im': [],
                    'chunk_loc': [],
                    'distance': []})
seg_angles = []
sample_ids = []
far_files = []
f = open(dat_file, 'r')
for line in f.readlines():
    if line.startswith('Image') and not line.endswith('x\n') and not line.endswith('x'):
        vals = line.split(',')
        seg_file = os.path.join(seg_folder, vals[1])

        # read in the segmented images
        seg_im = io.imread(seg_file)
        hole_bool = (seg_im != 2).ravel()
        b_val = 2
        print(np.unique(seg_im))
        if acf_angle:
        # plot the segmented image acfs with the orientation calculated from that
            seg_im[seg_im != 2] = 0
            seg_acf = xy_autocorr(seg_im)
            ang_acf = orientation(seg_acf)['angle'][0]
            if ang_acf < 0:
                interc = 300
            elif ang_acf > math.pi/2 and ang_acf < math.pi:
                interc = 300
            else:
                interc = 200
            pl.imshow(seg_im, origin='lower')
            abline(ang_acf, interc)
            pl.title(vals[1].replace('_', ', '))
            save_file = vals[1].rstrip('.tif') + '_angle_acf_onimage.png'
            pl.savefig(os.path.join(save_folder, save_file))
            pl.show()
            seg_angle = ang_acf
        else:
            # plot the segmented images with their orientation
            seg_angle = seg_orientation(seg_im, hole_bool)
            pl.imshow(seg_im, origin='lower')
            angle_shifted = seg_angle+math.pi/2
            print(math.degrees(angle_shifted))
            if angle_shifted < 0:
                interc = 511
            elif angle_shifted > math.pi/2 and angle_shifted < math.pi:
                interc = 511
            else:
                interc = 0
            abline(angle_shifted, interc)
            pl.title(vals[1].replace('_', ', '))
            save_file = vals[1].rstrip('.tif') + '_angle.png'
            pl.savefig(os.path.join(save_folder, save_file))
            pl.show()
            seg_angle = angle_shifted

        # read in the data image
        im_file = os.path.join(im_folder, vals[0])
        im = io.imread(im_file)[0]

        # split the image into chunks using info from the segmented image
        # uses shift = chunk size so no overlap
        chunks = split_image(im=im, seg_im=seg_im, bijel_val=b_val,
                             chunk_x=chunk_size, chunk_y=chunk_size,
                             shift=chunk_size)
        dat = dat.append(chunks)
        sample_id = vals[2]
        far_file = vals[3]
        for i in range(chunks.shape[0]):
            seg_angles.append(seg_angle)
            sample_ids.append(sample_id)
            far_files.append(far_file)
        # print(math.degrees(seg_angle))
f.close()

dat = dat.reset_index(drop=True)
dat = dat.astype({'chunk_im': 'object',
                  'chunk_loc': 'object',
                  'distance': 'float64'})

# add calculated hole angles to dataframe
dat['sample_id'] = sample_ids
dat['far_files'] = far_files
dat['seg_angle'] = seg_angles

plot_chunks = False

# calculate orientation of each chunk and add to dataframe
av_orientation = []
e_ratio = []
# label = []
for i in range(dat.shape[0]):
    im = dat.loc[i, 'chunk_im']
    sample_id = dat.loc[i, 'sample_id']
    seg_ang = dat.loc[i, 'seg_angle']
    acf = xy_autocorr(im)
    orient = orientation(acf)
    ang = get_angle(orient)
    av_orientation.append(ang)
    e_val_ratio = np.mean(orient['e_ratio'])
    e_ratio.append(e_val_ratio)
    ang_shifted = ang+math.pi/2
    if plot_chunks:
        pl.imshow(im, origin='lower')
        abline(seg_angle, 61)
        save_file = 'chunks/' + sample_id + '_' + str(chunk_size) + '_chunk' + str(i) + '.png'
        pl.savefig(os.path.join(save_folder, save_file))
        pl.show()
        # label.append(input('Is the image near (n), far (f) or unsure (x)?: '))

dat['av_or'] = av_orientation
dat['e_ratio'] = e_ratio
# dat['label'] = label

# calculate the angle relevant to the hole orientation and turn to degrees
dat['angle_deg'] = (dat['av_or']-dat['seg_angle'])*180/math.pi
dat['abs_angle'] = np.abs(dat['angle_deg'])

# look at distribution of eigenvalue ratio variable
pl.scatter(dat['distance'], dat['e_ratio'])
pl.show()

# try looking just at data with higher eigenvalue ratio
dat_reduced = dat[dat['e_ratio']>5]
x = dat_reduced['distance'].values
y = dat_reduced['abs_angle'].values
hist, xedges, yedges = np.histogram2d(x, y, bins=(40, 20))
angle_bin_means = []
for i in range(len(xedges)):
    if i != 0:
        xmin = xedges[i-1]
        xmax = xedges[i]
        # print(str(xmin)+', '+str(xmax))
        # print(xmax-xmin)
        dat_bins = dat_reduced[(dat_reduced['distance'] > xmin) & (dat_reduced['distance'] < xmax)]
        angle_bins = np.abs(dat_bins['abs_angle'])
        angle_bin_means.append(angle_bins.mean())

pl.scatter(xedges[:-1], angle_bin_means)
pl.xlabel('Distance from bubble trace (lower limit of bin)')
pl.ylabel('Magnitude of orientation angle')
pl.show()

pl.scatter(dat_reduced['distance'], np.abs(dat_reduced['abs_angle']))
pl.show()


# look at including different images
dat['sample_id'].unique()
id_to_include = [
                 '70_73',
                 '70_81',
                 '70_86',
                 '84_92',
                 '84_95',
                 # '70_96',
                 # '82_98',
                 '82_99',
                 '70_102'
                 ]

dat_new = dat[dat['sample_id'].isin(id_to_include)]

x = dat_new['distance'].values
y = dat_new['abs_angle']
n_bins = round(dat_new['distance'].max()/chunk_size*4)
hist, xedges = np.histogram(x, bins=n_bins)
angle_bin_means = []
angle_bin_std = []
bin_members = []
for i in range(len(xedges)):
    if i != 0:
        xmin = xedges[i-1]
        xmax = xedges[i]
        # print(str(xmin)+', '+str(xmax))
        # print(xmax-xmin)
        dat_bins = dat_new[(dat_new['distance'] > xmin) & (dat_new['distance'] < xmax)]
        angle_bins = dat_bins['abs_angle']
        angle_bin_means.append(angle_bins.mean())
        bin_members.append(len(angle_bins))
        angle_bin_std.append(angle_bins.std())

pl.scatter(xedges[:-1], bin_members)
pl.xlabel('Distance from bubble trace (lower limit of bin)')
pl.ylabel('Number of items in bin')
pl.title(str(id_to_include).replace('_', ', '))
pl.ylim(0)
pl.show()

bins_rem = -3

pl.scatter(xedges[:bins_rem-1], angle_bin_means[:bins_rem])
pl.fill_between(xedges[:bins_rem-1],
                (np.array(angle_bin_means[:bins_rem])-np.array(angle_bin_std[:bins_rem])),
                (np.array(angle_bin_means[:bins_rem])+np.array(angle_bin_std[:bins_rem])),
                color='b', alpha=.1)
pl.xlabel('Distance from bubble trace (lower limit of bin)')
pl.ylabel('mean of angle magnitude')
pl.title(str(id_to_include).replace('_', ', '))
pl.show()

pl.scatter(xedges[:bins_rem-1], angle_bin_means[:bins_rem])
pl.xlabel('Distance from bubble trace (lower limit of bin)')
pl.ylabel('mean of angle magnitude')
pl.title(str(id_to_include).replace('_', ', '))
pl.show()

pl.scatter(xedges[:-1], angle_bin_means[:])
pl.xlabel('Distance from bubble trace (lower limit of bin)')
pl.ylabel('mean of angle magnitude')
pl.title(str(id_to_include).replace('_', ', '))
pl.show()

dat_new.to_pickle(os.path.join(save_folder, 'new_dat.pkl'))

pd.DataFrame({'distance': xedges[:bins_rem-1],
              'ang': angle_bin_means[:bins_rem],
              'ang_err': angle_bin_std[:bins_rem]})

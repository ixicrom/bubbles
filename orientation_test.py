from bubble_tools import xy_autocorr, split_image
from skimage import io
import os
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import seaborn as sb
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# import imageio
# from sklearn.preprocessing import MinMaxScaler

dat_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/dat_list_73.csv'
im_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/'
seg_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/model_73'

# %% functions to calculate orientation

def seg_orientation(seg_im_array):
    # pl.imshow(seg_im_array)
    # pl.colorbar()
    # pl.show()
    im_x, im_y = np.meshgrid(np.arange(seg_im_array.shape[1]),
                             np.arange(seg_im_array.shape[0]))
    im_table = np.vstack((im_x.ravel(), im_y.ravel(), seg_im_array.ravel())).T
    im_table
    im_df = pd.DataFrame(im_table, columns=['x', 'y', 'val'])
    im_df['val'].unique()

    pixels = im_df[im_df['val'] == 0][['x', 'y']].values.T

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


# %% reading files and opening images


dat_liquid = pd.DataFrame({'chunk_im': [],
                           'chunk_loc': [],
                           'distance': []})
dat_particle = pd.DataFrame({'chunk_im': [],
                             'chunk_loc': [],
                             'distance': []})
dat_seg = []
f = open(dat_file, 'r')
for line in f.readlines():
    if not line.endswith('x') and line.startswith('Image'):
        vals = line.split(',')

        im_file = os.path.join(im_folder, vals[0])
        liquid_im = io.imread(im_file)[0]

        seg_file = os.path.join(seg_folder, vals[1])
        seg_im = io.imread(seg_file)[0]


        liquid = split_image(liquid_im, seg_im, 64, 64, 32)
        dat_liquid = dat_liquid.append(liquid)
        seg_angle = seg_orientation(seg_im)
        for i in range(liquid.shape[0]):
            dat_seg.append(seg_angle)

f.close()
dat_liquid.shape
len(dat_seg)
dat_liquid = dat_liquid.reset_index(drop=True)
dat_liquid = dat_liquid.astype({'chunk_im': 'object',
                                'chunk_loc': 'object',
                                'distance': 'float64'})
dat_liquid['seg_angle'] = dat_seg
print(dat_liquid.head())

# %% calculate ACF for each image (just looking at liquid channel)
dat_acf = []
# i = 0
for im in dat_liquid['chunk_im']:
    acf = xy_autocorr(im)
    dat_acf.append(acf)
    # imageio.imwrite(str(i)+'test.png', acf)
    # i += 1

dat_liquid['acf'] = dat_acf

print(dat_liquid.head())



# %% use function
test = orientation(dat_acf[0])
test

av_orientation = []
e_ratio = []
for im in dat_liquid['acf']:
    orient = orientation(im)
    av_orientation.append(np.mean(orient['angle']))
    e_ratio.append(np.mean(orient['e_ratio']))

dat_liquid['av_or'] = av_orientation
dat_liquid['e_ratio'] = e_ratio

dat_liquid.head()

# dat_liquid.plot(x='distance', y='av_or', style='o')
# pl.show()

# %% plot results
dat_plot = dat_liquid.sort_values('e_ratio').replace([np.inf, -np.inf],
                                                     np.nan).dropna()

dat_plot['angle_deg'] = (dat_plot['av_or']-dat_plot['seg_angle'])*180/math.pi

pl.scatter(dat_plot['distance'],
           dat_plot['angle_deg'],
           c=dat_plot['e_ratio'])
pl.xlabel('distance')
pl.ylabel('angle (degrees)')
pl.colorbar()
pl.show()

sb.scatterplot(data=dat_plot, x='distance', y='angle_deg',
               hue='e_ratio', alpha=0.7)


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

ang_200px_abs = np.abs(dat_plot[(dat_plot['distance']>175) & (dat_plot['distance']<225)]['angle_deg'])
ang_200px_abs.hist()
pl.xlabel('Magnitude of angle (deg)')
pl.ylabel('Frequency')
pl.title('Distance = 200 pixels (+/- 25px)')
pl.show()

ang_300px_abs = np.abs(dat_plot[(dat_plot['distance']>275) & (dat_plot['distance']<325)]['angle_deg'])
ang_300px_abs.hist()
pl.xlabel('Magnitude of angle (deg)')
pl.ylabel('Frequency')
pl.title('Distance = 300 pixels (+/- 25px)')
pl.show()

from bubble_tools import seg_orientation, split_image, orientation, xy_autocorr, get_angle
from skimage import io
import math
import numpy as np
import matplotlib.pyplot as pl
import matplotlib
matplotlib.style.core.reload_library()
pl.style.use('thesis')


def abline(angle_rad, intercept):
    slope = math.tan(angle_rad)
    axes = pl.gca()
    x_vals = np.array(axes.get_xlim())+0.5
    y_vals = intercept + slope * x_vals
    pl.plot(x_vals, y_vals, '--', color='r')


chunk_size = 64
im_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/Image73.lsm'
seg_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/region_growing/Classified_image73.tif'

im = io.imread(im_file)[0]
seg_im = io.imread(seg_file)
hole_bool = (seg_im != 2).ravel()

chunks = split_image(im=im, seg_im=seg_im, bijel_val=2, chunk_x=chunk_size, chunk_y=chunk_size, shift=chunk_size)

seg_angle = seg_orientation(seg_im, hole_bool)
seg_acf = xy_autocorr(seg_im)
seg_angle = orientation(seg_acf)['angle'][0]
seg_angle

seg_save = 'Image73'

pl.imshow(seg_im)
abline(seg_angle, 300)
pl.xlim(0, 512)
pl.savefig(seg_save + '_image.png')
pl.show()

pl.imshow(seg_acf)
abline(seg_angle, 300)
pl.xlim(0, 512)
pl.savefig(seg_save + '_acf.png')
pl.show()

# for i in range(chunks.shape[0]):
#     chunk_eg = chunks.loc[i, 'chunk_im']
#
#
#     # angle_alt = orientation(chunk_eg, n_bins=3)['angle'][1]
#     # angle_alt
#
#     acf = xy_autocorr(chunk_eg)
#
#     # pl.imshow(chunk_eg)
#     # abline(angle_alt, 30)
#     # pl.title(str(i))
#     # pl.ylim(chunk_size, 0)
#     # pl.xlim(0, chunk_size)
#     # pl.show()
#     ori = orientation(acf, n_bins=5, plot=True)
#     angle = get_angle(ori)
#     print(math.degrees(angle))
#
#     pl.imshow(acf)
#     abline(angle+math.pi/2, 30)
#     pl.title(str(i))
#     pl.ylim(chunk_size, 0)
#     pl.xlim(0, chunk_size)
#     pl.show()
#
#     pl.imshow(chunk_eg)
#     abline(angle+math.pi/2, 30)
#     pl.title(str(i))
#     pl.ylim(chunk_size, 0)
#     pl.xlim(0, chunk_size)
#     pl.show()

chunk_eg = chunks.loc[22, 'chunk_im']
dist_eg = chunks.loc[22, 'distance']

eg_save = 'Image73_chunk22_dist-' + str(dist_eg)

acf = xy_autocorr(chunk_eg)
ori = orientation(acf, n_bins=5, plot=True)
angle = get_angle(ori)
print(math.degrees(angle))

pl.imshow(chunk_eg)
abline(angle+math.pi/2, 30)
pl.ylim(chunk_size-0.5, 0)
pl.xlim(0, chunk_size-0.5)
pl.savefig(eg_save + '_image.png')
pl.show()

math.degrees(angle)

pl.imshow(acf)
abline(angle+math.pi/2, 30)
pl.ylim(chunk_size-0.5, 0)
pl.xlim(0, chunk_size-0.5)
pl.savefig(eg_save + '_acf.png')
pl.show()


# examples of data and chunk_size
new_im = np.where(seg_im != 2, -100, im)

pl.imshow(new_im)
pl.savefig('Image73_seg-cut.png')
pl.show()

print(chunks.loc[0, 'chunk_loc'])
print(chunks.loc[0, 'distance'])

pl.imshow(chunks.loc[0, 'chunk_im'])
pl.savefig('Image73_chunk0.png')
pl.show()

pl.imshow(xy_autocorr(chunks.loc[0, 'chunk_im']))
pl.savefig('Image73_acf0.png')
pl.show()

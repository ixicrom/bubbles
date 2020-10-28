from scipy import fft
from skimage import io
import numpy as np
import matplotlib.pyplot as pl
from bubble_tools import split_image

# %% variable setup
test_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/Image73.lsm'
seg_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/model_73/Classified_image73.tif'

im = io.imread(test_file)[0]

seg_im = io.imread(seg_file, as_gray=True)[0]

im_list, cent_list, d_list = split_image(im, seg_im, 128, 128, 64, plot_ims=False)


io.imshow(im_list[0])
pl.show()


# %% define functions
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


# %% try out functions
for i in range(len(im_list)):
    image = im_list[i]
    y_size, x_size = image.shape
    acf = xy_autocorr(image)
    acf_x = x_profile(acf)
    acf_y = y_profile(acf)

    pl.imshow(image)
    pl.title('Original image, distance = '+str(d_list[i]))
    pl.colorbar()
    pl.show()

    pl.contour(acf)
    pl.title('ACF contour, distance = '+str(d_list[i]))
    pl.colorbar()
    pl.show()

    pl.plot(acf_x[x_size//2+1:], label='x')
    pl.plot(acf_y[y_size//2+1:], label='y')
    pl.title('ACF mean, distance = '+str(d_list[i]))
    pl.legend()
    pl.show()

    # pl.plot([*range(x_size//2+1)],
    #         acf_x[:x_size//2+1],
    #         label='x')
    # pl.plot([*range(x_size//2+1, x_size//2+y_size//2)],
    #         acf_y[y_size//2+1:],
    #         label='y')
    # pl.title(str(d_list[i]))
    # pl.legend()
    # pl.show()

    pl.plot(acf[x_size//2+1:, x_size//2+1], label='x')
    pl.plot(acf[y_size//2+1, y_size//2+1:], label='y')
    pl.title('ACF centre, distance = '+str(d_list[i]))
    pl.legend()
    pl.show()

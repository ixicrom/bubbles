from scipy import fft
from skimage import io
import numpy as np
import matplotlib.pyplot as pl
from bubble_tools import split_image
import pandas as pd

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


# %% functions to extract variables from ACF
def turnpoints(lst):
    x = np.array(lst)
    n = len(x)
    x0 = x[0]-1.
    x2 = np.concatenate(([x0], x[:-1]))
    diffs = x != x2
    uniques = x[diffs]
    uniques
    n2 = len(uniques)
    poss = np.arange(n)[diffs]
    exaequos = np.concatenate((poss[1:n2], [n+1]))-poss-1
    '''
    at some point need to add in if statements to catch when
    things are wrong as with the R package
    '''
    m = n2-2
    vals = np.concatenate((np.arange(m)+2, np.arange(m)+1, np.arange(m)))
    ex = np.array(uniques[vals])
    ex = np.reshape(ex, (-1, m))
    ex = np.transpose(ex)
    peaks = [False]
    pits = [False]
    for i in range(m):
        peaks.append(ex[i, 1] == max(ex[i, ]))
        pits.append(ex[i, 1] == min(ex[i, ]))
    peaks.append(False)
    pits.append(False)
    tpts = [a or b for a, b in zip(peaks, pits)]
    if sum(tpts) == 0:
        tppos = np.nan
        peaks = [False]*n2
        pits = [False]*n2
    else:
        tppos = (poss+exaequos)[tpts]
    return tppos


def acf_variables(acf_x, acf_y):
    x_size = len(acf_x)
    y_size = len(acf_y)

    acf_x = acf_x[x_size//2+1:]
    acf_y = acf_y[y_size//2+1:]

    grad_x = (acf_x[4] - acf_x[0])/5
    grad_y = (acf_y[4] - acf_y[0])/5
    grad_diff = grad_x - grad_y

    tp_x = turnpoints(acf_x)[0]
    tp_y = turnpoints(acf_y)[0]
    tp_diff = tp_x - tp_y

    output = {'grad_x': grad_x,
              'grad_y': grad_y,
              'grad_diff': grad_diff,
              'tp_x': tp_x,
              'tp_y': tp_y,
              'tp_diff': tp_diff}
    return pd.Series(output)


# %% try out functions
var_list = []
for i in range(len(im_list)):
    image = im_list[i]
    y_size, x_size = image.shape
    acf = xy_autocorr(image)
    acf_x = x_profile(acf)
    acf_y = y_profile(acf)

    vars = acf_variables(acf_x, acf_y)
    dist = pd.Series({'distance': d_list[i]})
    vars = vars.append(dist)
    var_list.append(vars)

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
pd.concat(var_list, axis=1)

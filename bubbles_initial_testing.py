from skimage import io
import matplotlib.pyplot as pl
import scipy.fftpack as fftim
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def radial_profile(data, centre):
    y, x = np.indices((data.shape))
    r = np.sqrt((x-centre[0])**2+(y-centre[1])**2)
    r = r.astype(np.int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin/nr
    return radialprofile


def correlate(x, y):
    fr = fftim.fft2(x)
    fr2 = fftim.fft2(np.flipud(np.fliplr(y)))
    m, n = fr.shape
    cc = np.real(fftim.ifft2(fr*fr2))
    cc = np.roll(cc, int(-m/2+1), axis=0)
    cc = np.roll(cc, int(-n/2+1), axis=1)
    return cc


test_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/1280um/sample70/Image81.lsm'

test_im = io.imread(test_file)

io.imshow(test_im[0])


io.imshow(test_im[1])

test_a = test_im[0]

test_a[:100, :300].shape

io.imshow(test_a[:100, :300])

n_x = 4
n_y = 4

size_x = int(test_a.shape[1]/n_x)
size_y = int(test_a.shape[0]//n_y)

size_x

scaler = MinMaxScaler()
half_size = int(size_x/2)
im_list = []
for i in range(n_x):
    for j in range(n_y):
        im_tile = test_a[i*size_y:(i+1)*size_y, j*size_x:(j+1)*size_y]
        im_tile = scaler.fit_transform(im_tile)
        im_list.append(im_tile)
        io.imshow(im_tile)
        pl.title('Image ('+str(i)+', '+str(j)+')')
        pl.savefig('Image'+str(i)+'-'+str(j))
        pl.show()

        b = im_tile
        ac3 = correlate(b, b)
        autoCorr = radial_profile(ac3,
                                  (b.shape[0]/2,
                                   b.shape[1]/2
                                   )
                                  )[:256]/radial_profile(ac3,
                                                         (b.shape[0]/2.,
                                                          b.shape[1]/2.
                                                          )
                                                         )[:half_size][0]
        radProf = radial_profile(b,
                                 (b.shape[0]/2,
                                  b.shape[1]/2
                                  )
                                 )[:256]/radial_profile(b,
                                                        (b.shape[0]/2.,
                                                         b.shape[1]/2.
                                                         )
                                                        )[:half_size][0]
        pl.plot(autoCorr)
        pl.title('autoCorr ('+str(i)+', '+str(j)+')')
        pl.savefig('autoCorr'+str(i)+'-'+str(j))
        pl.show()

        pl.plot(radProf)
        pl.title('radProf ('+str(i)+', '+str(j)+')')
        pl.savefig('radProf'+str(i)+'-'+str(j))
        pl.show()

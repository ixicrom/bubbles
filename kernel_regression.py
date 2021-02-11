import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as pl
import pandas as pd
from bubble_tools import xy_autocorr
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib
matplotlib.style.core.reload_library()
pl.style.use('thesis')

im_folder = '/Users/s1101153/Dropbox/Emily+Paul meetings/Bubble Data'
dat_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/plots/rg_tuning/new_dat.pkl'
chunk_size = 32
chunk_y = chunk_size*4
chunk_x = chunk_size*4
shift = chunk_size*4

dat = pd.read_pickle(dat_file)

acf_list = []
for im in dat['chunk_im']:
    acf = xy_autocorr(im)
    acf.shape
    acf.ravel().shape
    acf_list.append(acf.ravel())

dat['acf'] = acf_list

from sklearn.preprocessing import MinMaxScaler

X = np.asarray(acf_list)
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
X_scaled = scaling.transform(X)


y = dat['distance'].values
X_scaled

# compare train/test split with and without stratify
min = np.amin(y)
max = np.amax(y)
n_bins_eg = 8
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)
pl.hist(y, bins=n_bins_eg)
pl.title('All data distribution')
pl.xlabel('Distance')
pl.ylabel('Frequency')
pl.savefig('hist_all.png')
pl.show()

pl.hist(y_train, bins=n_bins_eg)
pl.title('Training data distribution')
pl.xlabel('Distance')
pl.ylabel('Frequency')
pl.savefig('hist_train.png')
pl.show()

pl.hist(y_test, bins=n_bins_eg)
pl.title('Test data distribution')
pl.xlabel('Distance')
pl.ylabel('Frequency')
pl.savefig('hist_test.png')
pl.show()

bins = np.linspace(start=min, stop=max, num=n_bins_eg)

y_binned = np.digitize(y, bins, right=True)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0, stratify=y_binned)
pl.hist(y, bins=n_bins_eg)
pl.title('All data distribution')
pl.xlabel('Distance')
pl.ylabel('Frequency')
pl.savefig('hist_all_strat.png')
pl.show()

pl.hist(y_train, bins=n_bins_eg)
pl.title('Training data stratified distribution')
pl.xlabel('Distance')
pl.ylabel('Frequency')
pl.savefig('hist_train_strat.png')
pl.show()

pl.hist(y_test, bins=n_bins_eg)
pl.title('Test data stratified distribution')
pl.xlabel('Distance')
pl.ylabel('Frequency')
pl.savefig('hist_test_strat.png')
pl.show()

# try different sizes of bins
for num_bins in range(4, 10):
    bins = np.linspace(start=min, stop=max, num=num_bins)
    y_binned = np.digitize(y, bins, right=True)


    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                        test_size=0.3,
                                                        random_state=0,
                                                        stratify=y_binned)

    pl.hist(y, bins=num_bins)
    pl.title('All data distance distribution')
    pl.show()

    pl.hist(y_train, bins=num_bins)
    pl.title('Training data distance distribution')
    pl.show()

    pl.hist(y_test, bins=num_bins)
    pl.title('Test data distance distribution')
    pl.show()


    # try cross-validation to choose gamma and C
    tuned_parameters = [{'kernel' : ['rbf'],
                         'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
                         'C': [1, 10, 100, 1000, 10000]}]  # ,
                        # {'kernel': ['linear'],
                         # 'C': [1, 10, 100, 1000, 10000]}]

    clf = GridSearchCV(SVR(), tuned_parameters, scoring='r2', cv=5)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    means = clf.cv_results_['mean_test_score']
    print(means)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_all = clf.predict(X_scaled)

    fig, axes = pl.subplots(nrows=1, ncols=3, sharey=True)
    axes[0].scatter(y, y_pred_all, alpha=0.5)
    axes[0].xlabel = 'True distance'
    axes[0].plot(y, y, color='red', alpha=0.5)
    axes[1].scatter(y_train, y_pred_train, alpha=0.5)
    axes[1].scatter(y_train[clf.best_estimator_.support_],
                    y_pred_train[clf.best_estimator_.support_],
                    facecolor='none', edgecolor='red', alpha=0.3)
    axes[1].plot(y_train, y_train, color='red', alpha=0.5)
    axes[1].xlabel = 'True distance'
    axes[2].scatter(y_test, y_pred_test, alpha=0.5)
    axes[2].plot(y_test, y_test, color='red', alpha=0.5)
    axes[2].xlabel = 'True distance'
    pl.tight_layout()
    pl.show()

np.arange(len(y_train))

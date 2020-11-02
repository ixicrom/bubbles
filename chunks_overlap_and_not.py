from bubble_tools import split_image, x_profile, y_profile, xy_autocorr, acf_variables
from skimage import io
import os
import pandas as pd
import matplotlib.pyplot as pl
import sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.tree import DecisionTreeRegressor, plot_tree

dat_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/dat_list_73.csv'
im_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/'
seg_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/confocal_data/tunnels/model_73'

# %% reading files and opening images

dat_separate = pd.DataFrame({'chunk_im': [],
                             'chunk_loc': [],
                            'distance': []})
dat_overlap = pd.DataFrame({'chunk_im': [],
                            'chunk_loc': [],
                            'distance': []})
f = open(dat_file, 'r')
for line in f.readlines():
    if not line.endswith('x') and line.startswith('Image'):
        vals = line.split(',')

        im_file = os.path.join(im_folder, vals[0])
        im = io.imread(im_file)[0]

        seg_file = os.path.join(seg_folder, vals[1])
        seg_im = io.imread(seg_file, as_gray=True)[0]

        overlap = split_image(im, seg_im, 128, 128, 64)
        dat_overlap = dat_overlap.append(overlap)

        separate = split_image(im, seg_im, 128, 128, 128)
        dat_separate = dat_separate.append(separate)
f.close()

dat_overlap = dat_overlap.reset_index(drop=True)
dat_overlap = dat_overlap.astype({'chunk_im': 'object',
                                  'chunk_loc': 'object',
                                  'distance': 'float64'})
print(dat_overlap.describe())


dat_separate = dat_separate.reset_index(drop=True)
dat_separate = dat_separate.astype({'chunk_im': 'object',
                                    'chunk_loc': 'object',
                                    'distance': 'float64'})
print(dat_separate.describe())


# %% getting variables for separate-chunk data
acf_sep_list = []
for im in dat_separate['chunk_im']:
    acf = xy_autocorr(im)
    acf_x = x_profile(acf)[65:]
    acf_y = y_profile(acf)
    acf_sep_list.append(acf_variables(acf_x, acf_y))
acf_separate = pd.concat(acf_sep_list, axis=1).transpose()

dat_separate_all = pd.concat([dat_separate, acf_separate],
                             sort=False,
                             axis=1)

# print(dat_separate_all.describe())


# %% getting variables for overlapping-chunk data
acf_over_list = []
for im in dat_overlap['chunk_im']:
    acf = xy_autocorr(im)
    acf_x = x_profile(acf)[65:]
    acf_y = y_profile(acf)
    acf_over_list.append(acf_variables(acf_x, acf_y))
acf_overlap = pd.concat(acf_over_list, axis=1).transpose()
dat_overlap_all = pd.concat([dat_overlap, acf_overlap], sort=False, axis=1)

# %% look at data
print(dat_overlap_all.describe())
print(dat_separate_all.describe())

dat_overlap_vars = dat_overlap_all.drop(['chunk_im', 'chunk_loc'], axis=1)
dat_overlap_vars.hist()

dat_separate_vars = dat_separate_all.drop(['chunk_im', 'chunk_loc'], axis=1)
dat_separate_vars.hist()

# %% remove entries with outliers in any column
std_dev = 3.

dat_overlap_vars = dat_overlap_vars.dropna()
dat_overlap_vars = dat_overlap_vars[(np.abs(stats.zscore(dat_overlap_vars)) < float(std_dev)).all(axis=1)]
dat_overlap_vars.hist()

dat_separate_vars = dat_separate_vars.dropna()

dat_separate_vars = dat_separate_vars[(np.abs(stats.zscore(dat_separate_vars)) < float(std_dev)).all(axis=1)]
dat_separate_vars.hist()


# %% try some linear regression

X_sep = dat_separate_vars.drop('distance', axis=1)
y_sep = dat_separate_vars['distance']

model = LinearRegression()
scores_sep = []
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
for i, (train, test) in enumerate(kfold.split(X_sep, y_sep)):
    model.fit(X_sep.iloc[train, :], y_sep.iloc[train])
    scores_sep.append(model.score(X_sep.iloc[test, :], y_sep.iloc[test]))
print(np.mean(scores_sep))


X_over = dat_overlap_vars.drop('distance', axis=1)
y_over = dat_overlap_vars['distance']
scores_over = []
for i, (train, test) in enumerate(kfold.split(X_over, y_over)):
    model.fit(X_over.iloc[train, :], y_over.iloc[train])
    scores_over.append(model.score(X_over.iloc[test, :], y_over.iloc[test]))
print(np.mean(scores_over))

# %% different methods of cross validation:
model_norm = LinearRegression(normalize=True)
lin_scores_sep = cross_val_score(model_norm, X_sep, y_sep, cv=10)
print(np.mean(lin_scores_sep))

lin_scores_over = cross_validate(model_norm,
                                 X_over,
                                 y_over,
                                 cv=10,
                                 return_estimator=True)
print(np.mean(lin_scores_over['test_score']))

[i.coef_ for i in lin_scores_over['estimator']]

# %% try some decision trees

regressor = DecisionTreeRegressor(random_state=123)
tree_scores_sep = cross_val_score(regressor, X_sep, y_sep, cv=10)
print(np.mean(tree_scores_sep))

tree_scores_over = cross_validate(regressor,
                                  X_over,
                                  y_over,
                                  cv=10,
                                  return_estimator=True)
print(np.mean(tree_scores_over['test_score']))

for tr in tree_scores_over['estimator']:
    plot_tree(tr)
    pl.show()

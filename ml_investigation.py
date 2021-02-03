from bubble_tools import xy_autocorr, x_profile, y_profile, acf_variables
import pandas as pd
from skimage import io
import os
import numpy as np
import matplotlib.pyplot as pl
from sklearn.linear_model import LogisticRegressionCV
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import seaborn as sb
from sklearn.preprocessing import StandardScaler

im_folder = '/Users/s1101153/Dropbox/Emily+Paul meetings/Bubble Data'
dat_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/bubbles/plots/choose_parameters/data_rg_32.pkl'
chunk_size = 32
chunk_y = chunk_size*4
chunk_x = chunk_size*4
shift = chunk_size*4

dat = pd.read_pickle(dat_file)

# far_ims = dat['far_file'].unique()
# far_ims
# plot_ims = True
# far_chunks = []
# for im_file in far_ims:
#     print(im_file)
#     im_list = []
#     id_list = []
#     im = io.imread(os.path.join(im_folder, im_file))[1]
#     size_x = im.shape[1]
#     size_y = im.shape[0]
#
#     n_y_shift = (size_y-chunk_y)//shift+1
#     n_x_shift = (size_x-chunk_x)//shift+1
#
#     x_min = 0
#     x_max = chunk_x
#     for i in range(n_y_shift):
#         y_min = 0
#         y_max = chunk_y
#         for j in range(n_x_shift):
#             im_tile = im[y_min:y_max, x_min:x_max]
#             im_list.append(im_tile)
#             id_list.append(im_file)
#             if plot_ims:
#                 pl.imshow(im_tile)
#                 pl.show()
#
#             y_min += shift
#             y_max += shift
#         x_min += shift
#         x_max += shift
#         far_chunks.append(pd.DataFrame([im_list, id_list]))
#
# far_chunks_df = pd.concat(far_chunks, axis=1).transpose().reset_index()
#
# far_chunks_df = far_chunks_df.rename(columns={0: 'far_chunks', 1: 'far_file'})
#
# # calculate variables from acf of each set of image chunks and append to the df
# vars_far = []
# for im in far_chunks_df['far_chunks']:
#     acf = xy_autocorr(im)
#     middle = im.shape[0]//2+1
#     acf_x = x_profile(acf)[middle:]
#     acf_y = y_profile(acf)[middle:]
#     vars_far.append(acf_variables(acf_x, acf_y))
# vars_far_df = pd.concat(vars_far, axis=1).transpose()
# vars_far_df['label'] = ['f']*vars_far_df.shape[0]
# dat_far = pd.concat([far_chunks_df, vars_far_df], axis=1)
#
vars = []
for im in dat['chunk_im']:
    acf = xy_autocorr(im)
    middle = im.shape[0]//2+1
    acf_x = x_profile(acf)[middle:]
    acf_y = y_profile(acf)[middle:]
    vars.append(acf_variables(acf_x, acf_y))
vars = pd.concat(vars, axis=1).transpose()
dat = pd.concat([dat, vars], axis=1)


# define which of the original images are near and which are unknown
dat_near = dat[dat['distance'] <= chunk_size]
dat_near['label'] = ['n']*dat_near.shape[0]
dat_unknown = dat[dat['distance'] > chunk_size]
# add the next part if I use the far data from the original images
dist_90pc = np.percentile(dat['distance'], 90)
dat_unknown = dat_unknown[dat_unknown['distance'] < dist_90pc]
dat_far = dat[dat['distance'] > dist_90pc]
dat_far['label'] = ['f']*dat_far.shape[0]

dat_near.describe()
dat_far.describe()

dat_ml = pd.concat([dat_near, dat_far])
dat_ml.describe()
dat_ml
dat_ml['label']
for col in dat_ml.columns[5:14]:
    sb.boxplot(data=dat_ml,
               x='label',
               y=col)
    pl.show()
    pl.scatter(dat['distance'], dat[col])
    pl.xlabel('distance')
    pl.ylabel(col)
    pl.show()

variables = ['distance', 'angle_deg']
X = dat_ml.dropna().loc[:, variables]
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
y = dat_ml.dropna()['label']
y.describe()
print(19/36)

# try KNN
knn_cv = KNeighborsClassifier()

k_grid = {'n_neighbors': np.arange(1, 25)}
knn_gscv = GridSearchCV(knn_cv, k_grid, cv=5)

knn_fit = knn_gscv.fit(Xs, y)
knn_fit.predict(Xs)

print(knn_gscv.best_params_)
print(knn_gscv.best_score_)

# use knn classifier on rest of data:
for_pred_df = dat_unknown.dropna().loc[:, variables]
for_pred = scaler.transform(for_pred_df)
unknown_preds = knn_gscv.predict(for_pred)
unknown_preds

unknown_results = dat_unknown.dropna()
unknown_results.shape
unknown_results['label'] = unknown_preds

dat_unknown['label'] = unknown_results['label']

sb.swarmplot(data=dat_unknown,
             x='label',
             y='distance',
             size=10)
pl.title('KNN')
pl.show()

# try a decision tree
# regressor = DecisionTreeRegressor(random_state=123)
# tree_scores = cross_validate(regressor, X, y, cv=10, return_estimator=True)
# print(np.mean(tree_scores['test_score']))
# for tr in tree_scores['estimator']:
#     plot_tree(tr)
#     pl.show()


# try logistic regression
lr = LogisticRegressionCV(cv=5, class_weight='balanced')
lr_fit = lr.fit(Xs, y)
print(X.columns.values)
print(lr_fit.coef_)
print(lr.score(Xs, y))

lr_fit.predict(for_pred)

prob = lr_fit.predict_proba(for_pred)
lr_fit.classes_

prob_df = pd.DataFrame(prob, index = for_pred_df.index, columns = ['prob_f', 'prob_n'])
prob_df

prob_df['distance']=dat.loc[prob_df.index,'distance']
prob_df['abs_angle']=dat.loc[prob_df.index,'abs_angle']

prob_df

pl.scatter(prob_df['distance'], prob_df['prob_n'])
pl.xlabel('distance')
pl.ylabel('prob of being affected by bubble')
pl.show()

pl.scatter(prob_df['abs_angle'], prob_df['prob_n'])
pl.xlabel('absolute angle from bubble trace')
pl.ylabel('prob of being affected by bubble')
pl.show()

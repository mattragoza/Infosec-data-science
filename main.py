import matplotlib
matplotlib.use('Agg')
import sys, os, re
from operator import itemgetter
import numpy as np
import scipy as sp
import pandas as pd
pd.set_option('display.width', 150)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

import sklearn.pipeline as skl_pipeline
import sklearn.preprocessing as skl_preproc
import sklearn.feature_extraction as skl_featext
import sklearn.linear_model as skl_linear
import sklearn.model_selection as skl_modsel
import sklearn.metrics as skl_metrics


def plot_dist(plot_file, x):
    # plot distribution of user reputation
    bins = np.arange(0, 6, 1)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.distplot(np.log10(x), bins=bins, kde=False, ax=ax)
    ax.set_xticks(bins)
    ax.set_xticklabels('1e{}'.format(b) for b in bins)
    ax.set_xlabel('Reputation')
    ax.set_ylabel('# Users')
    text = 'min  = {}\nmax = {}'.format(x.min(), x.max())
    ax.text(0.025, 0.975, text, verticalalignment='top', transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(plot_file, bbox_inches='tight')


def plot_corr(plot_file, x, y):
    # plot correlation of true and predicted user reputation
    fig, ax = plt.subplots(figsize=(4,4))
    sns.regplot(x=x, y=y, data=users, ax=ax, fit_reg=False,
                scatter_kws=dict(alpha=0.25, s=2.5))
    ax.set_xlabel('True Reputation')
    ax.set_ylabel('Predicted Reputation')
    xlim = (0, 5000)
    ylim = (-1000, 6000)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.hlines(0, *ax.get_xlim(), linewidth=1.0)
    ax.vlines(0, *ax.get_ylim(), linewidth=1.0)
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', linewidth=0.5)
    mae = skl_metrics.mean_absolute_error(x, y)
    rmse = np.sqrt(skl_metrics.mean_squared_error(x, y))
    text = 'MAE  = {:.3f}\nRMSE = {:.3f}'.format(mae, rmse)
    ax.text(0.025, 0.975, text, verticalalignment='top', transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(plot_file, bbox_inches='tight')


# read data files and create additional user features
users = pd.read_csv('csv/Users.csv', index_col=0)
badges = pd.read_csv('csv/Badges.csv', index_col=0)
posts = pd.read_csv('csv/Posts.csv', index_col=0)
comments = pd.read_csv('csv/Comments.csv', index_col=0)

str_len = lambda x: len(x) if isinstance(x, str) else 0
time_since = lambda x: (pd.to_datetime('today') - dt.datetime.strptime(x,
                        '%Y-%m-%dT%H:%M:%S.%f')).total_seconds()

users['DisplayNameLength'] = users['DisplayName'].apply(str_len)
users['AboutMeLength'] = users['AboutMe'].apply(str_len)
users['HasWebsite'] = ~users['WebsiteUrl'].isnull()
users['HasProfileImage'] = ~users['ProfileImageUrl'].isnull()
users['HasAboutMe'] = users['AboutMeLength'] > 0
users['TimeSinceCreate'] = users['CreationDate'].apply(time_since)
users['TimeSinceAccess'] = users['LastAccessDate'].apply(time_since)

user_badges = badges.groupby(['UserId'])
users['NumBadges'] = user_badges.size()
users['NumUniqueBadges'] = user_badges['Name'].nunique()

user_posts = posts.groupby(['OwnerUserId'])
users['NumPosts'] = user_posts.size()
users['MeanPostScore'] = user_posts['Score'].mean()
users['MeanPostViews'] = user_posts['ViewCount'].mean()
users['MeanPostFavorites'] = user_posts['FavoriteCount'].mean()
users['MeanPostComments'] = user_posts['CommentCount'].mean()

user_comments = comments.groupby(['UserId'])
users['NumComments'] = user_comments.size()
users['MeanCommentScore'] = user_comments['Score'].mean()

users['AboutMe'].fillna('', inplace=True)
users.fillna(0, inplace=True)

numeric_cols = [
    'HasProfileImage',
    'HasWebsite',
    'HasAboutMe',
    'TimeSinceCreate',
    'TimeSinceAccess',
    'Views',
    'UpVotes',
    'DownVotes',
    'NumBadges',
    'NumUniqueBadges',
    'NumPosts',
    'MeanPostScore',
    'MeanPostViews',
    'MeanPostFavorites',
    'MeanPostComments',
    'NumComments',
    'MeanCommentScore'
]
feature_cols = numeric_cols + ['AboutMe']

model = skl_pipeline.Pipeline([
    ('feat', skl_pipeline.FeatureUnion([
        ('num', skl_pipeline.Pipeline([
            ('get', skl_preproc.FunctionTransformer(itemgetter(numeric_cols), validate=False)),
            ('poly', skl_preproc.PolynomialFeatures()),
            ('std', skl_preproc.StandardScaler()),
        ])),
        ('text', skl_pipeline.Pipeline([
            ('get', skl_preproc.FunctionTransformer(itemgetter('AboutMe'), validate=False)),
            ('tfidf', skl_featext.text.TfidfVectorizer()),
        ]))
    ])),
    ('reg', skl_linear.ElasticNet(alpha=1.0, l1_ratio=1.0))
])

param_grid = [dict(
    feat__num__poly__degree=[1, 2, 3],
    feat__num__poly__interaction_only=[True, False],
    feat__text__tfidf__max_df=[0.25, 0.5, 1.0],
    feat__text__tfidf__norm=[None, 'l1', 'l2'],
    reg__alpha=[1e-2, 1e-1, 1e0, 1e1, 1e2],
    reg__l1_ratio=[1.0, 0.5, 0.0],
)]

def root_mean_squared_error(*args, **kwargs):
    return np.sqrt(skl_metrics.mean_squared_error(*args, **kwargs))

scoring = dict(
    mae=skl_metrics.make_scorer(skl_metrics.mean_absolute_error),
    rmse=skl_metrics.make_scorer(root_mean_squared_error)
)

grid_search = skl_modsel.GridSearchCV(model, param_grid=param_grid, scoring=scoring,
                                      refit=False, verbose=10)

#users = users.loc[:100]
X = users.loc[:, feature_cols]
y = users.loc[:, 'Reputation']

grid_search.fit(X, y)
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.sort_values(by='mean_test_mae', inplace=True)
cv_results.to_csv('cv_results.csv')

best_params = cv_results.loc[0, 'params']
model.set_params(**best_params)
model.fit(X, y)
yp_train = model.predict(X)

users['TrainPredReputation'] = yp_train

plot_dist('true_dist.png', users['Reputation'])
plot_dist('train_pred_dist.png', users['TrainPredReputation'])
plot_corr('train_corr.png', users['Reputation'], users['TrainPredReputation'])

k_fold = skl_modsel.StratifiedKFold(n_splits=3,
                                    shuffle=True,
                                    random_state=0)

for train_idx, test_idx in k_fold.split(X, np.log10(y).astype(int)):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    yp_test = model.predict(X_test)

    users.loc[users.index[test_idx], 'TestPredReputation'] = yp_test

plot_dist('test_pred_dist.png', users['TestPredReputation'])
plot_corr('test_corr.png', users['Reputation'], users['TestPredReputation'])

print('done')

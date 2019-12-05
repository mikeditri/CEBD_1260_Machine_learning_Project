#Parameters for lightGBM classification
import os
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# make a directory to hold plot files
os.makedirs('plots', exist_ok=True)

model_lgb = LGBMClassifier(
        n_jobs=4,
        n_estimators=100000,
        boost_from_average='false',
        learning_rate=0.02,
        num_leaves=64,
        num_threads=4,
        max_depth=7,
        tree_learner = "serial",
        feature_fraction = 0.7,
        bagging_freq = 5,
        bagging_fraction = 0.5,
#         min_data_in_leaf = 75,
#         min_sum_hessian_in_leaf = 50.0,
        silent=-1,
        verbose=-1,
        device='cpu',
        )

#Parameters for RFC classification
clf = RandomForestClassifier(n_estimators=1000, max_depth=7,random_state=0,max_leaf_nodes=64,verbose=1,n_jobs=-1)

# import OneHotEncoder & define it
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories = 'auto',sparse=True)

kf = KFold(n_splits=5, random_state=10, shuffle=True)


def master_pipe(X_ohe,y):

  # place holder for k-fold scores
    scores = []

  # to differentiate files names produced by plt.savefig
    n = 1

  # model pipeline calculates model score and saves feature importance graph as .png file
    for i,(tr_idx, val_idx) in enumerate(kf.split(X_ohe,y)):
        print('Fold :{}'.format(i))
        tr_X = X_ohe[tr_idx]  # training for this loop
        tr_y = y[tr_idx] #
        val_X = X_ohe[val_idx]# validation data for this loop
        val_y = y[val_idx]
        # here build your models
        model = model_lgb
        model.fit(tr_X, tr_y, eval_set=[(tr_X, tr_y), (val_X, val_y)], eval_metric = 'auc', verbose=100,
                  early_stopping_rounds= 50)
        #picking best model?
        pred_val_y = model.predict_proba(val_X,num_iteration=model.best_iteration_)[:,1]
        #measuring model vs validation
        score = roc_auc_score(val_y,pred_val_y)
        scores.append(score)
        print('current performance by auc:{}'.format(score))
        lightgbm.plot_importance(model, ax=None, height=0.2, xlim=None, ylim=None, title='Feature importance',
                                 xlabel='Feature importance', ylabel='Features', importance_type='split',
                                 max_num_features=20, ignore_zero=True, figsize=None, grid=True, precision=3)
        # in python plots dir will be auto-created
        #plt.show()
        plt.savefig('plots/feature_importance{}.png'.format(n), format='png')
        plt.close()
        n=n+1

def master_pipe_RFC(X_ohe,y):
    # place holder for k-fold scores
    scores_rfc = []

  # model pipeline calculates model score and saves feature importance graph as .png file
    for i,(tr_idx, val_idx) in enumerate(kf.split(X_ohe,y)):
        print('Fold :{}'.format(i))
        tr_X = X_ohe[tr_idx]  # training for this loop
        tr_y = y[tr_idx] #
        val_X = X_ohe[val_idx]# validation data for this loop
        val_y = y[val_idx]

        # here we build the model
        model = clf
        model.fit(tr_X, tr_y)
        #picking best model?
        pred_val_y = model.predict(val_X)
        #measuring model vs validation
        score_rfc = roc_auc_score(val_y,pred_val_y)
        scores_rfc.append(score_rfc)
        print('current performance by auc:{}'.format(score_rfc))

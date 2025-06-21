#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
import plt_params

# Load the data : lucas_test.csv and lucas_train.csv
train = pd.read_csv('../data/lucas_train.csv')
x_train = train.drop('Lung_cancer', axis=1).astype(float)
y_train = train['Lung_cancer'].astype(float)

# %%

logistic_cv = LogisticRegressionCV( Cs=100, cv=5, penalty='l1', solver='liblinear', random_state=12345, scoring='neg_log_loss' )
n_smpl = 500
sample = train.sample( n_smpl )
logistic_cv.fit( sample.drop('Lung_cancer', axis=1), sample['Lung_cancer'] )
selected_features = x_train.columns[logistic_cv.coef_[0] != 0]
paths = logistic_cv.coefs_paths_[1][0,:,:] # first fold
lambdas = 1 / logistic_cv.Cs_
loss = np.mean( logistic_cv.scores_[1], axis=0 )

plt.figure( figsize=(10,6) )
for i in range( len(selected_features) ):
    plt.plot( lambdas, paths[:,i], label=selected_features[i] )
plt.xscale('log')
plt.legend(loc='upper right')
plt.xlabel('Regularisation')
plt.ylabel('Coefficient')
plt.savefig('../figures/lasso_coef_path.png')

plt.figure(tight_layout=True)
plt.plot( lambdas, loss )
plt.axvline( 1/logistic_cv.C_[0], color='red', label='Selected' )
plt.xscale('log')
plt.xlabel('Regularisation')
plt.ylabel('Negative log loss')
plt.legend()
plt.savefig('../figures/lasso_likelihood_path.png')

# %%


def get_counts( data, n_rep, n_smpl ):
    lasso_counts = np.zeros( 11 )
    n_folds = 5
    df = data
    for i in range(n_rep):
        if n_rep < len(data):
            df = data.sample( n_smpl ) # maybe we can test with replacement

        model = LogisticRegressionCV( Cs=10, cv=n_folds, penalty='l1', solver='liblinear', random_state=12345, scoring='neg_log_loss' )
        res = model.fit( df.drop('Lung_cancer', axis=1), df['Lung_cancer'] )
        
        non_zero = np.where( res.coef_ != 0, True, False )[0]
        lasso_counts[non_zero] += 1
    frac = lasso_counts / n_rep
    trial_var = frac*(1-frac)
    lasso_std = trial_var / np.sqrt(n_rep)
    return frac, lasso_std

n_rep = 100
n_var = train.columns.size - 1
n_n_smpl = np.arange( 100, 800, 100 )
n_n_smpl = np.concatenate( (n_n_smpl, np.arange( 800, 1000, 50 )) )
tot_lasso_counts = np.zeros( (len(n_n_smpl), n_var) )
tot_lasso_std_counts = np.zeros( (len(n_n_smpl), n_var) )
for i, n_smpl in enumerate(n_n_smpl):
    lasso_counts, lasso_std = get_counts( train, n_rep, n_smpl ) 
    tot_lasso_counts[i,:] = lasso_counts
    tot_lasso_std_counts[i,:] = lasso_std

fig, axs = plt.subplots( 1, 1, figsize=(10, 6), layout='tight' )
for i in range(11):
    axs.errorbar( n_n_smpl, tot_lasso_counts[:,i], yerr=tot_lasso_std_counts[:,i], label=x_train.columns[i], capsize=5 )

axs.legend(loc='lower right')
axs.set_xlabel('Number of samples')
axs.set_ylabel('Frequency of selection')
plt.savefig('../figures/lasso-trainsize-sensitivity.png')

# %%

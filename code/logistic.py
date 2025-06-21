# %% 
import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import Logit
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tools.eval_measures import bic
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import plt_params

train = pd.read_csv('../data/lucas_train.csv')
test = pd.read_csv('../data/lucas_test.csv')
x_train = train.drop('Lung_cancer', axis=1).astype(float)
y_train = train['Lung_cancer'].astype(float)
x_test = test.drop('Lung_cancer', axis=1).astype(float)
y_test = test['Lung_cancer'].astype(float)
x_train = sm.add_constant( x_train )
x_test = sm.add_constant( x_test )

# print the proprotions of every class in the training set
l = ''
for col in train.columns:
    print( f'{col} : {train[col].mean():.3}' )


#%% 

def loglike( y, p ):
    return np.sum( y*np.log(p) + (1-y)*np.log(1-p) )


def print_model_res( model, out_df, model_label='Model', print_summary=False ):
    if print_summary:
        print( model.summary() )
    prob_test = model.predict( x_test )
    auc_ = roc_auc_score(y_test, prob_test)
    bic_ = bic( loglike(y_test,prob_test), len(test), model.df_model )
    print( model_label, '\n BIC:', bic_, '\n AUC:', auc_, '\n' )
    dic = {'Model': model_label}
    dic['AUC'] = f'{auc_:.4f}' 
    dic['BIC'] = f'{bic_:.2f}' 
    out_df = out_df._append( dic, ignore_index=True )
    return out_df

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    # taken from https://stackoverflow.com/a/39358752
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax

# %%
import scipy.stats as stats
# 3.45e-4 and 5.17e-14 and 0.273

def llr_p_value( model, nested_model ):
    lr = 2*(model.llf - nested_model.llf)
    p = stats.chi2.cdf(lr, model.df_model - nested_model.df_model)
    print(f'{1-p:.3}')
    return 1-p

column_names = ['Model', 'AUC', 'BIC']
param_df = pd.DataFrame( columns=column_names )

true_model = smf.logit( 'Lung_cancer ~ Smoking + Coughing + Fatigue + Genetics + Allergy', data=train ).fit()
param_df = print_model_res( true_model, param_df, model_label='True' )

logistic = Logit(y_train,x_train).fit()
param_df = print_model_res( logistic, param_df, model_label='Total' )

llr_p_value( logistic, true_model )

no_fatigue_model = smf.logit( 'Lung_cancer ~ Smoking + Coughing + Genetics + Allergy', data=train ).fit()
param_df = print_model_res( no_fatigue_model, param_df, model_label='No fatigue' )
# compute the likelihood ratio test
llr_p_value( true_model, no_fatigue_model )

no_allergy_model = smf.logit( 'Lung_cancer ~ Smoking + Coughing + Genetics + Fatigue', data=train ).fit()
param_df = print_model_res( no_allergy_model, param_df, model_label='No allergy' )
llr_p_value( true_model, no_allergy_model )


fig, ax = render_mpl_table(param_df, header_columns=0, col_width=3.0)
plt.savefig('../figures/regression-results.png')



# %%

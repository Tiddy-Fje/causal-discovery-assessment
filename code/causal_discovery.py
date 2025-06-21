import pandas as pd
import numpy as np
import rpy2.robjects.packages as rpackages
import matplotlib.pyplot as plt
import plt_params

bn = rpackages.importr('bnlearn')
utils = rpackages.importr('utils')

x = pd.read_csv("..\\data\\lucas_train.csv")

def compute_std_on_mean( counts, n_rep ): 
    frac = np.array( [counts[lab] / n_rep for lab in counts] )
    trial_var = frac*(1-frac)
    sum_var = trial_var / n_rep
    return np.sqrt(sum_var) 

def get_mass( data, n_rep, n_smpl ):
    col = 12*['factor']
    iamb_counts = {lab: 0 for lab in data.columns}
    iamb_counts.pop('Lung_cancer')
    pc_counts = iamb_counts.copy()
    for i in range(n_rep):
        if n_rep < len(data):
            df = data.sample( n_smpl ) # maybe we can test with replacement
            df.to_csv('..\\data\\out.csv', index=False) 
        else:
            data.to_csv('..\\data\\out.csv', index=False)

        xx = utils.read_csv("..\\data\\out.csv", header=True, colClasses = col)
        y = bn.learn_mb(xx, node = "Lung_cancer", method = "iamb")
        yy = bn.pc_stable(xx)[1][11][0] # access mb of Lung_cancer
        for lab in y:
            iamb_counts[lab] += 1
        for lab in yy:
            pc_counts[lab] += 1
    
    iamb_std = compute_std_on_mean( iamb_counts, n_rep )
    pc_std = compute_std_on_mean( pc_counts, n_rep )
    iamb_counts = np.array([iamb_counts[lab] for lab in iamb_counts])
    pc_counts = np.array([pc_counts[lab] for lab in pc_counts])

    return iamb_counts/n_rep, iamb_std, pc_counts/n_rep, pc_std

n_rep = 100
n_var = x.columns.size - 1
n_n_smpl = np.arange( 100, 800, 100 )
n_n_smpl = np.concatenate( (n_n_smpl, np.arange( 800, 1000, 50 )) )
tot_iamb_mass = np.zeros( (len(n_n_smpl), n_var) )
tot_pc_mass = np.zeros( (len(n_n_smpl), n_var) )
tot_iamb_std = np.zeros( (len(n_n_smpl), n_var) )
tot_pc_std = np.zeros( (len(n_n_smpl), n_var) ) 
for i, n_smpl in enumerate(n_n_smpl):
    iamb_mass, iamb_std, pc_mass, pc_std = get_mass( x, n_rep, n_smpl ) 
    tot_iamb_mass[i,:] = iamb_mass
    tot_pc_mass[i,:] = pc_mass
    tot_iamb_std[i,:] = iamb_std
    tot_pc_std[i,:] = pc_std

# plot the counts +- standard deviation for each label as a function of n_smpl
fig, axs = plt.subplots( 2, 1, figsize=(10, 9), layout='tight', sharex=True)
for i in range(11):
    #print( n_n_smpl.size, tot_iamb_mass[:,i].shape )
    axs[0].errorbar( n_n_smpl, tot_iamb_mass[:,i], yerr=tot_iamb_std[:,i], label=x.columns[i], capsize=5 )
    axs[1].errorbar( n_n_smpl, tot_pc_mass[:,i], yerr=tot_pc_std[:,i], capsize=5 )
axs[0].legend(loc='upper right')
axs[1].set_xlabel('Number of samples')
axs[0].set_ylabel('Frequency of selection')
axs[1].set_ylabel('Frequency of selection')
plt.savefig('..\\figures\\mb-trainsize-sensitivity.png')


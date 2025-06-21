import matplotlib.pyplot as plt

## increasing font sizes for the figures
plt.rc( 'axes', titlesize=17 ) 
plt.rc( 'axes', labelsize=15 ) 
#plt.rc( 'lines', linewidth=2.2 ) 
plt.rc( 'xtick', labelsize=12 ) 
plt.rc( 'ytick', labelsize=12 )
plt.rc( 'legend',fontsize=12 )
# make layout tight
plt.rc( 'figure', tight_layout=True )
# same for subplots
plt.rc( 'subplots', tight_layout=True )

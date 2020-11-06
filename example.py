# -*- coding: utf-8 -*-
"""
Run a PLS - Discriminant Analysis on a set of variables and target variables
Romain Lafarguette, https://romainlafarguette.github.io/
Time-stamp: "2020-11-05 14:58:55 Romain"
"""

###############################################################################
#%% Modules and methods
###############################################################################
# Global modules
import pandas as pd                                     # Data management
import numpy as np                                      # Numeric tools
import statsmodels.api as sm                            # Statistics

# Functional imports
from plswrapper import PLS
from linearmodels import OLS

# Graphics
import matplotlib
matplotlib.use('TkAgg') # Must be called before importing plt
import matplotlib.pyplot as plt                         # Graphical package  
import seaborn as sns                                   # Graphical tools


# Pandas options
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 10)

# Set the style for all charts
sns.set(style='white', font_scale=3, palette='deep', font='serif',
        rc={'text.usetex' : False}) # LateX font

###############################################################################
#%% Import standard data
###############################################################################
# Dataset on US macro quarterly data
df = sm.datasets.macrodata.load_pandas().data.copy()

# Create a date index with end of quarter convention
dates_l = [f'{y:.0f}-Q{q:.0f}' for y,q in zip(df['year'], df['quarter'])]
df = df.set_index(pd.to_datetime(dates_l) + pd.offsets.QuarterEnd())

# Clean some variables
df['rgdp_growth'] = df['realgdp'].rolling(4).sum().pct_change(4)
df = df.rename(columns={'infl':'inflation', 'unemp':'unemployment'})

# New variables
df['rgdp_growth_fwd_1y'] = 100*df['rgdp_growth'].shift(-4)
df['realdpi_yoy'] = 100*df['realdpi'].pct_change(4)
df['cpi_yoy'] = 100*df['cpi'].pct_change(4)
df['realcons_yoy'] = 100*df['realcons'].pct_change(4)
df['realinv_yoy'] = 100*df['realinv'].pct_change(4)
df['realgovt_yoy'] = 100*df['realgovt'].pct_change(4)

df = df.dropna().copy()

# Normalize the dataframe
dfz = (df - df.mean())/df.std()

###############################################################################
#%% Compute a PLS
###############################################################################
y_l = ['rgdp_growth_fwd_1y']
x_l = ['realdpi_yoy', 'cpi_yoy', 'tbilrate', 'realint']

# x_l = ['realdpi_yoy', 'cpi_yoy', 'tbilrate', 'realint',
#        'realcons_yoy', 'realinv_yoy', 'realgovt_yoy']


# Fit 
pls_fit = PLS(y_l, x_l, data=dfz, num_vars='all')

# Compute the component
df['fci_pls'] = pls_fit.component

# Extract the loadings
pls_params = pls_fit.summary
pls_params['pls_params'] = pls_params['loadings'].copy()

###############################################################################
#%% OLS of: PLS ~ Components
###############################################################################
ols_fit = OLS(dfz['fci_pls'], dfz[x_l]).fit()
ols_params = ols_fit.params

pls_params.loc[ols_params.index, 'ols_params'] = ols_params.values

ds = pls_params.copy()

adj = ds.pls_params[0]/ds.ols_params[0]

ds['ols_params_adj'] = ds['ols_params']*adj

###############################################################################
#%% Plots of the coefficients
###############################################################################
fig, ax = plt.subplots(1, 1)
ds[['pls_params', 'ols_params']].plot.bar(ax=ax)
ax.axhline(y=0, color='k')
ax.legend(frameon=False)
ax.set_xticklabels(ds.index, rotation=0, fontsize=30)
ax.set_title('Comparison between PLS loadings and OLS coefficients')
plt.show()

###############################################################################
#%% OLS of: PLS ~ Components
###############################################################################

# Plot the quantities of interest
fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
for x in x_l:
    ax0.plot(df[x], label=x)
ax0.legend(frameon=False, fontsize=15)
ax0.set_title('X dataframe', y=1.02)

ax1.plot(df['fci_pls'], lw=3, color='firebrick', ls='-', label='FCI PLS')

ax1.set_title('PLS FCI', y=1.02)

ax2.plot(df['rgdp_growth_fwd_1y'], lw=3, color='navy', ls='-',
         label='US real GDP growth t+1y')
ax2.set_title('Future GDP Growth', y=1.02)

corrcoeff = 100*df[['rgdp_growth_fwd_1y', 'fci_pls']].corr().iloc[0,1]

plt.suptitle(f'Correlation FCI and Y: {corrcoeff:.0f}%')

fig.autofmt_xdate()

plt.plot()
plt.show()



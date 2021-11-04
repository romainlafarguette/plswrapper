# -*- coding: utf-8 -*-
"""
Run a PLS - Discriminant Analysis on a set of variables and target variables
Romain Lafarguette, https://romainlafarguette.github.io/
Time-stamp: "2021-11-03 20:26:35 RLafarguette"
"""

###############################################################################
#%% Modules and methods
###############################################################################
# Modules imports
import pandas as pd                                     ## Data management
import numpy as np                                      ## Numeric tools

import matplotlib.pyplot as plt

# Method imports
from sklearn.cross_decomposition import PLSRegression   ## PLS
from sklearn.preprocessing import scale

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

###############################################################################
#%% PLS Discriminant Analysis Class Wrapper
###############################################################################
class PLS(object):
    """ 
    Data reduction through PLS-discriminant analysis and variables selection 

    Inputs:
    - dep_vars: list; list of dependent variables
    - reg_vars: list; list of regressors variables
    - data: pandas df; data to train the model on
    - num_vars: 'all', integer; number of variables to keep, ranked by VIP
        if 'all': keep all the variables
    
    Output:
    - first_component : the first component of the PLS of the Xs reduction
    - output_frame = frame containing the variables and their transformation
    - summary_frame = frame with the results of the model (loadings, vip, R2)


    """
    __description = "Partial Least Squares with variables selection"
    __author = "Romain Lafarguette, IMF, rlafarguette@imf.org"

    #### Class Initializer
    def __init__(self, dep_vars, reg_vars, data, num_vars='all'):

        #### Attributes
        self.dep_vars = dep_vars
        self.reg_vars = reg_vars
        self.df = data.dropna(subset=self.dep_vars + self.reg_vars)

        # Scale all the variables (important)
        self.df = pd.DataFrame(scale(self.df.values), index=self.df.index,
                               columns=self.df.columns)
        
        ## Put parametrized regression as attribute for consistency
        self.pls1 = PLSRegression(n_components=1) 

        ## Unconstrained fit: consider all the variables 
        self.ufit = self.pls1.fit(self.df[self.reg_vars],
                                  self.df[self.dep_vars])

        ## Return the component and summary of the unconstrained model
        ## To save computation time, run it by default for both models        
        self.component_unconstrained = self.__component(self.ufit,
                                                        self.dep_vars,
                                                        self.reg_vars, self.df)

        self.target_unconstrained = self.__target(self.ufit,
                                                  self.dep_vars,
                                                  self.reg_vars, self.df)

        self.summary_unconstrained = self.__summary(self.ufit, self.dep_vars,
                                                    self.reg_vars, self.df)

        ## Variables selection
        if num_vars == 'all': # Unconstrained model: constrained is identical
            self.top_vars = self.reg_vars # The best variables are the full set
            self.fit = self.ufit
            self.component = self.component_unconstrained
            self.target = self.target_unconstrained
            self.summary = self.summary_unconstrained
            self.loadings = self.summary['loadings'] # Have them directly
            
        elif num_vars > 0: ## Constrained model
            self.num_vars = int(num_vars)
            
            ## Identify the most informative variables from the unconstrained
            self.top_vars = list(self.summary_unconstrained.sort_values(
                by=['vip'], ascending=False).index[:self.num_vars])

            ## Now run the constrained fit on these variables
            self.cfit = self.pls1.fit(self.df[self.top_vars],
                                      self.df[self.dep_vars])

            ## Return the main attributes, consistent names with unconstrained
            self.fit = self.cfit
            
            self.component = self.__component(self.cfit, self.dep_vars,
                                              self.top_vars, self.df)
            
            self.target = self.__target(self.cfit, self.dep_vars,
                                        self.top_vars, self.df)

            self.summary = self.__summary(self.cfit, self.dep_vars,
                                          self.top_vars, self.df)
                      
        else:
            raise ValueError('Number of variables parameter misspecified')

        
    #### Internal class methods (start with "__")
    def __vip(self, model):
        """ 
        Return the variable influence in the projection scores
        Input has to be a sklearn fitted model
        Not available by default on sklearn, so it has to be coded by hand
        """
        ## Get the score, weights and loadings
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        p, h = w.shape

        ## Initialize the VIP
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)

        for i in range(p):
            weight = [(w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h)]
            vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
        return(vips)

    def __summary(self, fit, dep_vars, reg_vars, df):
        """
        Return the summary information about the fit
        """
        
        ## Store the information into a pandas dataframe
        dr = pd.DataFrame(reg_vars, columns=['variable'], index=reg_vars)
        dr['loadings'] = fit.x_loadings_ # Loadings
        dr['vip'] = self.__vip(fit) ## Variable importance in projection
        dr['score'] = fit.score(df[reg_vars],df[dep_vars]) # Score
        
        ## Return the sorted summary frame
        return(dr.sort_values(by=['vip'], ascending=False))
    
    ## Write short ancillary functions to export the results into pandas series
    def __component(self, fit, dep_vars, reg_vars, df):
        """
        Return the first component of the fit
        """
        comp = fit.fit_transform(df[reg_vars], df[dep_vars])[0]
        comp_series = pd.Series(comp.flatten(), index=self.df.index)
        return(comp_series)

    def __target(self, fit, dep_vars, reg_vars, df):
        """
        Return the target of the fit (reduced in case of multiple variables)
        """
        target = fit.fit_transform(df[reg_vars], df[dep_vars])[1]
        target_series = pd.Series(target.flatten(), index=self.df.index)
        return(target_series)

    def predict(self, dcond=None):
        """ 
        Apply the dimension reduction learned on new predictors
        Input:
            - dcond: Pandas frame with the conditioning frame. 
            If none, use the in-sample prediction with the original data

        Output:
            - The prediction
 
        """

        # Extract the coefficients
        beta = self.loadings

        # Distinguish between in-sample and customized fit
        if dcond==None: # The fit is done in-sample, using data input         
            X = self.df[list(beta.index)]
        else:
            m = 'Conditioning frame should contain the predictors in columns'
            assert all(x in dcond.columns for x in beta.index),m
            X = scale(dcond.values) # Should always scale input variable
            X = pd.DataFrame(X, index=dcond.index, columns=dcond.columns)

        # Compute the projection as a simple matrix product
        dpred = pd.DataFrame(np.dot(X, beta), index=X.index,
                             columns=self.dep_vars)        
        return(dpred)

    def predict_plot(self, dcond=None):
        """ Prediction plots with raw variables, loadings, etc. """

        # Data
        dpred = self.predict(dcond)
        beta = self.loadings

        # Plots
        fig, axs = plt.subplots(1, 3)
        ax1, ax2, ax3 = axs.ravel()

        style_l = ['-', '--', '-.', ':']*3
        color_l = list('rgbkymc')*3
        
        # Raw variables
        for idx, var in enumerate(beta.index):
            ax1.plot(self.df.index, self.df[var], label=var, lw=2,
                     ls=style_l[idx], color=color_l[idx])
        ax1.xaxis.set_tick_params(labelsize='xx-small')
        ax1.legend(fontsize='xx-small', frameon=False, handlelength=1)
        ax1.set_title('Raw variables', y=1.02)

        # Loadings
        label_l = list(beta.index)
        for idx, var in enumerate(label_l):
            ax2.bar(var, beta[var], label=var, color=color_l[idx])
        ax2.legend(fontsize='xx-small', frameon=False, handlelength=1,
                   loc='upper right')
        ax2.set_xticks([])
        ax2.set_title('Loadings coefficients', y=1.02)
        
        # Output
        ax3.plot(dpred.index, dpred, label='Prediction', lw=2)
        ax3.xaxis.set_tick_params(labelsize='xx-small')
        ax3.legend(fontsize='small', frameon=False)
        ax3.set_title('Output', y=1.02)
        
        plt.subplots_adjust(left=0.05, right=0.95,  wspace=0.3)
        
        return(fig)
                  
###############################################################################
#%% Example
###############################################################################
if __name__ == '__main__':    
    from sklearn.datasets import make_regression     # Generate example data
    X, y = make_regression(n_samples=200, n_features=3,noise=4,random_state=42)
    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['y'] = y
    cond_vector = df.tail(1) # Example of a conditioning vector

    # Fit the model
    target = ['y']
    var_l = ['x1', 'x2', 'x3']
    pls_fit = PLS(target, var_l, df, num_vars='all')
    pls_fit.predict().head(5)
    # Should return:
    #           y
    # 0 -0.469897
    # 1  1.523225
    # 2  0.502270
    # 3  0.236980
    # 4 -0.007572


    # pls_fit.predict_plot()
    # plt.show()

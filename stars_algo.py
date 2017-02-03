# STARS Algorithm
# Rodionov's (2004) STARS algorithm for the mean
# Date: July 18, 2014
# Author: Paul Stey

import pandas as pd 
import numpy as np
from scipy.stats import t


## quick helper function that returns the sign (or zero)
def sign(x):
    if x > 0:
        result = 1
    elif x < 0:
        result = -1
    elif x == 0:
        result = 0
    return result


def stars_mean(X, xcol, date_col, p_val = .05, cut_len = 10):
    '''
    STARS algorithm (Rodionov, 2004) testing for mean shifts. `X` is a pandas 
    dataframe, `xcol` is the string with the column name for the variable of 
    interest, `date_col` is the column name for the variable with the date.
    '''    
    # Exclude rows with missing values on target variable
    X = X[pd.notnull(X[xcol])]

    x = X[xcol]                          # get vector of variable of interest
    n = len(x)

    temp = np.array([])
    var_tmp = 0

    ## get average SD over all cut-off lengths
    for j in range(n-cut_len+2):
        var_tmp = x[j:(j+cut_len)].var()
        temp = np.append(temp, var_tmp)

    sigma = np.sqrt(temp.mean())

    # Compute deviation representing significant shit
    t_crit = abs(t.ppf(p_val/2, 2*cut_len-2))
    diff = t_crit*sigma*np.sqrt(2/cut_len)  

    # Initialize objects to store shift information
    shft_points = pd.Series([])
    shft_rsi = pd.Series([])

    xbar = x[0:cut_len].mean()			# start calculation of moving average
    back_stepping = False				
    step_cnt = 0

    # Loop through each observation beyond burn-in period
    for i in range(cut_len, n):
        shft_found = False			    # begin by assuming not at shift point
        deviation = x.iat[i] - xbar     # deviation for this observation

        # Determine if we're potentially in regime shift
        if abs(deviation) > diff:
            n_obs = min(i+cut_len, n)   # num of observations to use in RSI calc

            # distance from crit. value for cut_len - 1 observations
            x_star = sign(deviation) * (x.iloc[i:n_obs] - xbar) - diff
            rsi = x_star.cumsum()/cut_len/sigma         # compute Regime Shift Index

            # if RSI stays positive, shift was found, begin new regime
            if not any(rsi < 0 ):
                shft_points[i] = X[date_col].iloc[i]	# save shift date
                shft_rsi[i] = rsi.iat[(len(rsi)-1)]		# save RSI value

                xbar = x.iloc[i:n_obs].mean()			# new regime's mean

                ## tell algorithm backstep until before new mean
                back_stepping = True
                step_cnt = 1

        # If no shift found, update moving average
        if not back_stepping:
            xbar = x.iloc[(i - cut_len + 1):(i+1)].mean()

        # If we're backstepping, update count
        if back_stepping and step_cnt < (cut_len - 1):
            step_cnt += 1
        else:
            back_stepping = False

    # Bind output into dataframe
    result = pd.concat([shft_points, shft_rsi], axis = 1)
    return result



import numpy as np
import pandas as pd
import scipy.special
import csv
# Load data
# Load data
returns = pd.read_csv("nasdaq_100_monthly_returns.csv")
returns.set_index('Date', inplace=True)
risk_free_rate = pd.read_csv("real_interest_rate.csv")
risk_free_rate.set_index('Date', inplace=True)

# Ensure both dataframes are aligned on the same dates
returns, risk_free_rate = returns.align(risk_free_rate, join='inner', axis=0)

# Convert data to numeric, forcing errors to NaN
returns = returns.apply(pd.to_numeric, errors='coerce')
risk_free_rate = risk_free_rate.apply(pd.to_numeric, errors='coerce')

# Combine the returns and risk-free rate for synchronized cleaning
combined_df = returns.copy()
combined_df['RiskFreeRate'] = risk_free_rate

# Check for NaN or infinite values in the combined dataframe
print("NaN values in combined_df:\n", combined_df.isna().sum())
print("Infinite values in combined_df:\n", np.isinf(combined_df).sum())

# Replace NaN and infinite values with 0
combined_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

# Separate the cleaned data back into returns and risk-free rate
cleaned_returns = combined_df.drop(columns=['RiskFreeRate']).to_numpy()
cleaned_rate = combined_df['RiskFreeRate'].to_numpy().reshape(-1, 1)

# Subtract risk-free rate from returns
R = cleaned_returns - cleaned_rate

# Calculate mean and covariance matrix
T = R.shape[0]
N = R.shape[1]

# Mean matrix (N x 1)
mu = np.mean(R, axis=0).reshape(-1, 1)

print(mu.shape)
R_centered = R - mu.T

sigma = 0.04 #using risk tolerance from the paper
# Calculate the sample covariance matrix
Sigma = (R_centered.T @ R_centered) / (T - 1) #N x N covariance matrix (unbiased T - 1 estimator)


#now we calculate parameters for learning

import scipy.special as sc
theta_s = mu.T @ np.linalg.inv(Sigma) @ mu #maximum squared sharpe ratio

#check section 1.5.2 for implementation.  Keeps squared sharpe psoitive
a = N/2
b = (T - N)/2
x = theta_s/(1 + theta_s) # Upper bound
B = sc.betainc(a, b, x)*sc.beta(a,b)#converting regularized to nonregularized incomplete beta function



theta_adj = ((T - N - 2)*theta_s - N)/T + 2*(theta_s**(N/2))*((1 + theta_s)**(-1*(T - 2)/2))/(T*B)
r_c = sigma*(1 + theta_adj)/np.sqrt(theta_adj)
r_star = (r_c*theta_adj)/(1 + theta_adj)

parameters = {"theta_s": theta_s[0][0], "theta_adj": theta_adj[0][0], "r_c": r_c[0][0], "r_star": r_star[0][0], "sigma": sigma}


Sigma =  pd.DataFrame(Sigma, columns = returns.columns, index = returns.columns)
Sigma.to_csv("nasdaq_Sigma_assets.csv")

mu =  pd.DataFrame(mu, columns = ["mean"], index = returns.columns)
mu.to_csv("nasdaq_mu_assets.csv")

R = pd.DataFrame(R, columns = returns.columns, index = returns.index)
R.to_csv("nasdaq_excess_returns_assets.csv")

print(parameters)
parameters = pd.DataFrame(parameters, index = [0])
parameters.to_csv("nasdaq_maxser_asset_parameters.csv")


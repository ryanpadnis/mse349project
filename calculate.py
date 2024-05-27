import numpy as np
import pandas as pd
import scipy.special
import csv
returns = pd.read_csv("RUSS2000_monthly_returns")

#need to estimate mean and covariance matrix using plug-in estimates
#then will use this to sample data and test estimators for maximum sharpe ratio/portoflio weights from paper



returns.set_index('Date', inplace=True)
risk_free_rate = pd.read_csv("risk_free_rate.csv")
rate = risk_free_rate.to_numpy()[:,1].reshape(-1,1)
returns_array = returns.to_numpy() #let's do everything in numpy for simplicity
excess_returns = returns_array - rate #broadcast together (vectorize)

R = excess_returns #going to set to R now
T =R.shape[0]
N = R.shape[1]
R = np.array(R, dtype = np.float64)
#note that all the parameters enlisted below are actually the estimates (subscript s is w.r.t the sample plug-in estimate)
mu = (np.sum(R, axis=0)/T) # N x 1 mean matrix
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

parameters = {"theta_s": theta_s, "theta_adj": theta_adj, "r_c": r_c, "r_star": r_star, "sigma": sigma}
with open('maxser_asset_parameters.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for key, value in parameters.items():
        writer.writerow([key, value])

Sigma =  pd.DataFrame(Sigma, columns = returns.columns, index = returns.columns)
Sigma.to_csv("Sigma_assets.csv")

mu =  pd.DataFrame(mu, columns = ["mean"], index = returns.columns)
mu.to_csv("mu_assets.csv")

R = pd.DataFrame(R, columns = returns.columns, index = returns.index)
R.to_csv("excess_returns_assets.csv")

parameters = pd.DataFrame(parameters, index = [0])
parameters.to_csv("maxser_asset_parameters.csv")
print(parameters)

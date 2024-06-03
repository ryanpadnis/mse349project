from scipy.linalg import sqrtm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import lars_path
from sklearn.metrics import mean_squared_error

# Step 1: Gather Data
#Yuze, be careful of the first column.  For some reason the first column is what the column names should be, use my following code to fix that up (same from earlier)
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.optimize import minimize

# Load data
mean = pd.read_csv("nasdaq_mu_assets.csv")
assets = mean.columns
mean.set_index("Unnamed: 0", inplace=True)
mean = mean.values.T[0]

cov = pd.read_csv("nasdaq_Sigma_assets.csv")
cov.set_index("Unnamed: 0", inplace=True)
Sigma = cov.values

R = pd.read_csv("nasdaq_excess_returns_assets.csv")
R.set_index("Date", inplace=True)
T = R.shape[0]
N = R.shape[1]
R = R.values

parameters = pd.read_csv("nasdaq_maxser_asset_parameters.csv")
r_c = parameters["r_c"].values[0]
r_c = r_c * np.ones(T)

risk_constraint = parameters["sigma"].values[0]

# Define the number of folds for cross-validation
num_folds = 10

# Define a list of alpha values to try




# Initialize the KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize lists to store average test MSE scores for each alpha
avg_test_mse_scores = []
# Iterate over the alpha values
zetas = []
for train_index, test_index in kf.split(R):
    X_train, X_test = R[train_index], R[test_index]
    y_train, y_test = r_c[train_index], r_c[test_index]
    # Initialize Lasso regression model
    alphas, _, coefs = lars_path(X_train, y_train)
    # Fit the model

    l1_norms = np.sum(np.abs(coefs), axis=0)
    # Plot alphas against the L1 norms

    risks = []
    for i in range(alphas.shape[0]):
        weights = coefs[:, i]
        risk = weights@Sigma@weights
        risks.append(risk)
    closest_risk_index = np.argmin(np.abs(np.array(risks) - risk_constraint))
    closest_alpha = alphas[closest_risk_index]
    zetas.append(closest_alpha)

    optimal_lambda = np.mean(zetas)

#find closest to optimal_lambda now, that is our portfolio
alphas, _, coefs = lars_path(R, r_c)
closest_alpha_index = np.argmin(np.abs(alphas - optimal_lambda))
closest_alpha = alphas[closest_alpha_index]
w_star = coefs[:, closest_alpha_index]
print(optimal_lambda)
plt.figure(figsize=(10, 6))
plt.plot(alphas, risks[:95])

plt.xlabel('Alpha')
plt.ylabel('Risk')
plt.title('Alpha vs Portfolio Risk')
plt.savefig('nasdaq_alphas.png')
plt.show()
print(w_star)
weighted = w_star.T@mean

expected_return = np.dot(w_star, mean)
# Compute portfolio variance
portfolio_variance = w_star.T @ Sigma @ w_star

# Compute portfolio standard deviation
portfolio_std_dev = np.sqrt(portfolio_variance)

# Calculate Sharpe Ratio
sharpe_ratio = (expected_return) / portfolio_std_dev

optimal = pd.DataFrame(w_star)
optimal.to_csv('nasdaq_optimal.csv')




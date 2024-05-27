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
mean = pd.read_csv("mu_assets.csv")
assets = mean.columns
mean.set_index("Unnamed: 0", inplace=True)
mean = mean.values.T[0]

cov = pd.read_csv("Sigma_assets.csv")
cov.set_index("Unnamed: 0", inplace=True)
Sigma = cov.values

R = pd.read_csv("excess_returns_assets.csv")
R.set_index("Date", inplace=True)
T = R.shape[0]
N = R.shape[1]
R = R.values

parameters = pd.read_csv("maxser_asset_parameters.csv")
r_c = parameters["r_c"].values[0]
r_c = r_c * np.ones(T)

risk_constraint = parameters["sigma"].values[0]

# Define the number of folds for cross-validation
num_folds = 10

# Define a list of alpha values to try
alphas = [0.002, 0.003,0.004,0.005,0.006,0.007,0.008]



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
    plt.figure(figsize=(10, 6))

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

weighted = w_star.T@mean

expected_return = np.dot(w_star, mean)
# Compute portfolio variance
portfolio_variance = w_star.T @ Sigma @ w_star

# Compute portfolio standard deviation
portfolio_std_dev = np.sqrt(portfolio_variance)

# Calculate Sharpe Ratio
sharpe_ratio = (expected_return) / portfolio_std_dev

optimal = pd.DataFrame(w_star)
optimal.to_csv('optimal.csv')





"""
old implementation using lasso, doesn't work though
# Step 3: Perform 10-fold Cross-Validation
for train_index, test_index in kf.split(R):
    X_train, X_test = R[train_index], R[test_index]
    y_train, y_test = r_c[train_index], r_c[test_index]
    # Step 4: Obtain the whole solution path for Lasso (alpha corresponds to lambda in the context)
    alphas = np.logspace(-4, 0, 100)  # Range of alphas (lambdas)
    best_zeta = None
    min_risk_diff = float('inf')

    for alpha in reversed(alphas):
        print(alpha)
        lasso = Lasso(alpha)
        lasso.fit(X_train, y_train)
        weights = lasso.coef_
        print(weights)
        breakpoint()

        # Calculate the portfolio risk (standard deviation of portfolio returns)
        portfolio_variance = weights.T @ np.cov(X_train.T) @ weights
        portfolio_risk = np.sqrt(portfolio_variance)

        # Calculate the difference between portfolio risk and the risk constraint
        risk_diff = abs(portfolio_risk - risk_constraint)

        if risk_diff < min_risk_diff:
            min_risk_diff = risk_diff
            best_zeta = np.sum(np.abs(weights)) / np.linalg.norm(weights, 1)

    zeta_values.append(best_zeta)

# Step 6: Compute the average zeta
optimal_zeta = np.mean(zeta_values)
print(f"Optimal zeta: {optimal_zeta}")

# Use optimal zeta to compute final portfolio weights using the whole dataset
final_alpha = None
min_risk_diff = float('inf')

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(R, r_c)
    weights = lasso.coef_

    portfolio_variance = weights.T @ Sigma @ weights
    portfolio_risk = np.sqrt(portfolio_variance)

    risk_diff = abs(portfolio_risk - risk_constraint)

    if risk_diff < min_risk_diff:
        min_risk_diff = risk_diff
        final_alpha = alpha

lasso_final = Lasso(alpha=final_alpha)
lasso_final.fit(R, r_c)
final_weights = lasso_final.coef_

print(f"Final weights: {final_weights}")
print(f"Final portfolio risk: {np.sqrt(final_weights.T @ Sigma @ final_weights)}")


"""
import numpy as np
import pandas as pd



parameters = pd.read_csv("maxser_asset_parameters.csv")
optimal_weights = pd.read_csv("optimal.csv")
R = pd.read_csv("excess_returns_assets.csv")
R.set_index("Date", inplace=True)
R = R.values
print(R)
optimal_weights.set_index("Unnamed: 0", inplace=True)
optimal_weights = optimal_weights.values
theta = parameters["theta_adj"].values[0]
sigma = parameters["sigma"].values[0]
theta_s = parameters["theta_s"].values[0]

mu = pd.read_csv("mu_assets.csv")
mu.set_index("Unnamed: 0", inplace=True)
N = mu.shape[0]
#consider following portfolios for baseline comparison
Sigma =pd.read_csv("Sigma_assets.csv")
Sigma.set_index("Unnamed: 0", inplace=True)

mu = mu.values.reshape(mu.shape[0])
Sigma = Sigma.values
print(np.sqrt(theta))
print(theta_s)
num_simulations = 1000
T = 240

def evaluate_portfolio(T, num_simulations,optimal_weights, mu, Sigma):
    portfolio_returns = np.zeros((num_simulations, T))

    # Perform simulations
    np.random.seed(1)  # For reproducibility
    for i in range(num_simulations):
        # Generate returns from a multivariate normal distribution
        simulated_returns = np.random.multivariate_normal(mu, Sigma, T)
        # Compute portfolio returns for this simulation
        portfolio_returns[i, :] = np.dot(simulated_returns, optimal_weights).reshape(T)

    sharpe_ratios = []

    # Compute the Sharpe ratio for each simulation
    for i in range(num_simulations):
        mean_return = np.mean(portfolio_returns[i, :])
        std_return = np.std(portfolio_returns[i, :])
        sharpe_ratio = mean_return / std_return
        sharpe_ratios.append(sharpe_ratio)

    # Convert to a numpy array for easier handling
    sharpe_ratios = np.array(sharpe_ratios)

    # Output the mean and standard deviation of the Sharpe ratios across simulations
    mean_sharpe_ratio = np.mean(sharpe_ratios)
    std_sharpe_ratio = np.std(sharpe_ratios)
    portfolio_risk = (optimal_weights.T @ Sigma @ optimal_weights)[0][0]

    return mean_sharpe_ratio, std_sharpe_ratio, portfolio_risk

one = np.ones(optimal_weights.shape)
kan_weights = (1/3)*((T - N - 1)*(T - N - 4)/(T* (T - 2)))*np.linalg.inv(Sigma)@mu
kan_weights = kan_weights.reshape(-1,1)



import pingouin as pg

# Example: Generating multivariate normal data

# Perform Mardia's test
mardia_test = pg.multivariate_normality(R, alpha=0.05)
print(mardia_test)


#explain here why this doesn't converge for russel 2000
"""kan_sharpe, std_sharpe_kan, kan_portfolio_risk = evaluate_portfolio(T, num_simulations, kan_weights, mu, Sigma)
optimal_sharpe, std_sharpe_optimal, optimal_portfolio_risk = evaluate_portfolio(T, num_simulations, optimal_weights, mu, Sigma)
benchmark_portfolio = ((sigma/np.sqrt(theta))*np.linalg.inv(Sigma)@mu).reshape(-1,1)#plugin using the adjusted theta
benchmark_sharpe, std_sharpe_benchmark, benchmark_portfolio_risk = evaluate_portfolio(T, num_simulations, benchmark_portfolio, mu, Sigma)
plugin_portfolio = ((sigma/np.sqrt(theta_s))*np.linalg.inv(Sigma)@mu).reshape(-1,1)
plugin_sharpe, std_sharpe_plugin, plugin_portfolio_risk = evaluate_portfolio(T, num_simulations, plugin_portfolio, mu, Sigma)
equal_portfolio = np.ones(optimal_weights.shape)/optimal_weights.shape[0]
equal_sharpe, std_sharpe_equal, equal_portfolio_risk = evaluate_portfolio(T, num_simulations, equal_portfolio, mu, Sigma)
print("Equal portfolio: ")
print(equal_sharpe)
print(equal_portfolio_risk)
print(std_sharpe_equal)
print("MAXSER portfolio: ")
print(optimal_sharpe)
print(optimal_portfolio_risk)
print(std_sharpe_optimal)
print("benchmark portfolio: ")#usign MAXSER as plugin
print(benchmark_sharpe)
print(benchmark_portfolio_risk)
print(std_sharpe_benchmark)
print("kan and zhao portfolio: ")
print(kan_sharpe)
print(kan_portfolio_risk)
print(std_sharpe_kan)
print("kan and zhao portfolio: ")#regulafr plugin theoretuical what markowitz says
print(plugin_sharpe)
print(plugin_portfolio_risk)
print(std_sharpe_plugin)"""


#kan and zhao three fund portfolio
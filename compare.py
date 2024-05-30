import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
R = pd.read_csv("excess_returns_assets.csv")
R.set_index("Date", inplace=True)
R = R.values


#now for the sp500 excess returns
risk_free_rate = pd.read_csv("real_interest_rate.csv")
rate = risk_free_rate.to_numpy()[:,1].reshape(-1,1)
data = pd.read_csv("sp500_monthly_returns")
data.set_index("Date", inplace=True)
sp500 = data.values - rate



def calculate_metrics(returns):
    # Mean return
    mean_return = np.mean(returns, axis=0)

    # Volatility (Standard Deviation)
    volatility = np.var(returns, axis=0)

    # Sharpe Ratio (assuming risk-free rate is 0 for simplicity)
    sharpe_ratio = mean_return / volatility

    # Cumulative Return
    cumulative_return = np.prod(returns + 1, axis=0) - 1

    # Maximum Drawdown
    cumulative_returns = np.cumprod(returns + 1, axis=0)
    peak = np.maximum.accumulate(cumulative_returns, axis=0)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown, axis=0)

    return mean_return, volatility, sharpe_ratio, cumulative_return, max_drawdown


# Calculate metrics for both datasets
mean_return1, volatility1, sharpe_ratio1, cumulative_return1, max_drawdown1 = calculate_metrics(sp500)
mean_return2, volatility2, sharpe_ratio2, cumulative_return2, max_drawdown2 = calculate_metrics(R)

# Normalize metrics (except for Sharpe Ratio)
mean_return_norm1 = mean_return1 / np.abs(mean_return1).max()
volatility_norm1 = volatility1 / np.abs(volatility1).max()
cumulative_return_norm1 = cumulative_return1 / np.abs(cumulative_return1).max()
max_drawdown_norm1 = max_drawdown1 / np.abs(max_drawdown1).max()

mean_return_norm2 = mean_return2 / np.abs(mean_return2).max()
volatility_norm2 = volatility2 / np.abs(volatility2).max()
cumulative_return_norm2 = cumulative_return2 / np.abs(cumulative_return2).max()
max_drawdown_norm2 = max_drawdown2 / np.abs(max_drawdown2).max()

# Plotting
metrics = ['Mean Return', 'Volatility', 'Sharpe Ratio', 'Cumulative Return', 'Maximum Drawdown']
norm_metrics1 = [mean_return_norm1, volatility_norm1, sharpe_ratio1, cumulative_return_norm1, max_drawdown_norm1]
norm_metrics2 = [mean_return_norm2, volatility_norm2, sharpe_ratio2, cumulative_return_norm2, max_drawdown_norm2]



# Find the minimum and maximum values across all volatilities
y_min = -0.05
y_max = 0.05

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(mean_return1)
plt.title('Normalized Mean Excess Return for SP500')
plt.xlabel('Asset')
plt.ylabel('Value')
plt.grid(True)
plt.ylim(y_min, y_max)  # Set y-axis limits

plt.subplot(2, 1, 2)
plt.plot(mean_return2)
plt.title('Normalized Mean Excess Return for Russ 2000')
plt.xlabel('Asset')
plt.ylabel('Value')
plt.grid(True)
plt.ylim(y_min, y_max)  # Set y-axis limits

plt.tight_layout()
plt.savefig('comparison_drawdown_plots')
plt.show()


"""plt.figure(figsize=(10, 12))
for i, metric in enumerate(metrics):
    plt.subplot(5, 2, 2 * i + 1)
    plt.plot(norm_metrics1[i])
    plt.title(f'Normalized {metric} for SP500')
    plt.xlabel('Asset')
    plt.ylabel('Value')
    plt.grid(True)

    plt.subplot(5, 2, 2 * i + 2)
    plt.plot(norm_metrics2[i])
    plt.title(f'Normalized {metric} for Russ 2000')
    plt.xlabel('Asset')
    plt.ylabel('Normalized Value')
    plt.grid(True)

plt.tight_layout()
plt.savefig('comparison_plots')
plt.show()"""


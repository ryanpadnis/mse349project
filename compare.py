import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
#mean_return1, volatility1, sharpe_ratio1, cumulative_return1, max_drawdown1 = calculate_metrics(sp500)
mean_return2, volatility2, sharpe_ratio2, cumulative_return2, max_drawdown2 = calculate_metrics(R)

"""# Normalize metrics (except for Sharpe Ratio)
mean_return_norm1 = mean_return1 / np.abs(mean_return1).max()
volatility_norm1 = volatility1 / np.abs(volatility1).max()
cumulative_return_norm1 = cumulative_return1 / np.abs(cumulative_return1).max()
max_drawdown_norm1 = max_drawdown1 / np.abs(max_drawdown1).max()"""

mean_return_norm2 = mean_return2 / np.abs(mean_return2).max()
volatility_norm2 = volatility2 / np.abs(volatility2).max()
cumulative_return_norm2 = cumulative_return2 / np.abs(cumulative_return2).max()
max_drawdown_norm2 = max_drawdown2 / np.abs(max_drawdown2).max()
sharpe_ratio2 = sharpe_ratio2/np.abs(sharpe_ratio2).max()

# Plotting
metrics = ['Mean Return', 'Volatility', 'Sharpe Ratio', 'Cumulative Return', 'Maximum Drawdown']
norm_metrics2 = [mean_return_norm2, volatility_norm2, sharpe_ratio2, cumulative_return_norm2, max_drawdown_norm2]
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
fig.subplots_adjust(hspace=0.5, wspace=0.3)
axs = axs.flatten()

for i, (metric, norm_metric) in enumerate(zip(metrics, norm_metrics2)):
    ax = axs[i]
    ax.plot(norm_metric)
    ax.set_title(metric)
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized Value')
    ax.legend()

# Hide the empty subplot in the last row
axs[-1].axis('off')


plt.tight_layout()
plt.savefig("nasdaq_plots")
plt.show()


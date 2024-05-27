# download bond pricing data, can infer risk free rate over this time period as a result
# use 20 year treasury bond.  Subrtact inflation rate from it
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# de-annualize yearly interest rates
#maybe jsut download one month treasury rate instead???? then subtract inflation rate and we're done
def deannualize(annual_rate, periods=12):
    return (1 + annual_rate) ** (1 / 12) - 1


def get_risk_free_rate():
    # download 3-month US Treasury bills rates (^IRX represents the 13-week T-Bill)
    annualized = yf.download("^IRX", start="2002-01-01", end="2022-01-01")["Adj Close"]
    # De-annualize the daily rates
    daily = annualized.apply(deannualize)

    # Create a dataframe with both annualized and daily rates
    return pd.DataFrame({"monthly": annualized, "daily": daily})


if __name__ == "__main__":
    # Resample to monthly frequency and calculate the mean
    rates = get_risk_free_rate().resample("M").last()/100
    monthly_rates = rates["monthly"]
    monthly_rates.to_csv("risk_free_rate.csv")
    # insert monthly inflation here
    CPI = pd.read_csv("CPI.csv")[1:241]
    CPI.set_index('DATE', inplace=True)
    CPI = CPI.pct_change()
    CPI = CPI.fillna(0.0)
    CPI.set_index(monthly_rates.index, inplace=True)

    real_interest_rate = monthly_rates.values.reshape(-1, 1) - CPI.values
    real_interest_rate = pd.DataFrame(real_interest_rate, columns = ["RATE"], index = monthly_rates.index)
    real_interest_rate.to_csv("real_interest_rate.csv")
    """
    plt.figure()
    plt.plot(real_interest_rate)
    plt.show()"""

    # Display the first few rows of the monthly rates

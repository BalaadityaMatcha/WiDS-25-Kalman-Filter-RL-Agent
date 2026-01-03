import yfinance as yf
import pandas as pd
import numpy as np

from pykalman import KalmanFilter

from scipy import stats
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns


# Part 1
'''
Use for downloading at first, uncomment the following lines
'''
# df = yf.download(
#     "MSFT",
#     start="2015-01-01",
#     end="2024-12-31",
#     progress=True
# )


# df.columns = df.columns.droplevel(1)
# df.to_csv("MSFT_2015_2024.csv")

df = pd.read_csv(
    "./MSFT_2015_2024.csv",
    index_col=0,
    parse_dates=True
)

# Renamed for easier usage later
df.rename(columns={"Close":"price","High":"high","Low":"low","Open":"open","Volume":"volume"}, inplace=True)


# Part 2 - Feature Engineering
price = df["price"]

# Moving Averages
mavg_20d = price.rolling(window = 20, center = False).mean()
mavg_60d = price.rolling(window = 60, center = False).mean()
std_60d = price.rolling(window = 60, center = False).std()

# Log returns and lagged returns
df["price_lag"] = df["price"].shift(1)
df["log_ret"] = np.log(df["price"]/df["price_lag"])
df["log_ret_lag"] = df["log_ret"].shift(1)

# ROC
df["roc_60"] = df["price"].pct_change(60)

# Rolling volatility measures
df["vol_20"] = df["log_ret"].rolling(20, center=False).std()

# Momentum
df["mom_20"] = mavg_20d - mavg_60d

# Some volume-based features
df["volume_avg_20"] = df["volume"].rolling(20, center=False).mean()
df["volume_ratio"] = df["volume"]/df["volume_avg_20"]
df["dollar_volume"] = df["price"] * df["volume"]

# Clearing NaNs
df = df.dropna()


# Part 3

def Kalman(x, y):
    n = len(y)
    # Observation Matrix: [1, x]
    obs_mat = np.vstack([np.ones(n), x]).T[:, np.newaxis]
    
    trans_cov = np.diag([1e-5, 1e-3])
    
    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=[0, 0],
        initial_state_covariance=np.eye(2),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=0.5,
        transition_covariance=trans_cov
    )
    
    # Run filter on the whole history
    state_means, _ = kf.filter(y.values)
    return state_means

# Applying to the whole dataframe at once to avoids the "reset" problem where the filter forgets context at the split point
states = Kalman(df["log_ret_lag"], df["log_ret"])

df["kalman_alpha"] = states[:, 0]
df["kalman_beta"]  = states[:, 1]

# Calculating Predictions & Errors for the whole history
df["kalman_pred"] = df["kalman_alpha"] + (df["kalman_beta"] * df["log_ret_lag"])
df["kalman_error"] = df["log_ret"] - df["kalman_pred"]


split = int(0.8 * len(df))
train_idx = df.index[:split]
test_idx  = df.index[split:]

# Part 4
df["fpr"] = df["price"].shift(-1) / df["price"] # Future price ratio

features = ["kalman_beta", "kalman_error", "vol_20", "roc_60"]

df_ml = df.dropna().copy() 
X = df_ml[features] # Predictors
y = df_ml["fpr"] # Responses

X_train = X.iloc[:split]
y_train = y.iloc[:split]

X_test = X.iloc[split:]
y_test = y.iloc[split:]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Merging prediction data into the real dataframe
df["pred_fpr"] = np.nan
df.loc[X_train.index, "pred_fpr"] = y_pred_train
df.loc[X_test.index, "pred_fpr"] = y_pred_test

# Part 5

# Generating Trading signals using the predicted ratios
e = 0.002
cond = [df["pred_fpr"] > 1 + e, df["pred_fpr"] < 1 - e]
df["pred_signal"] = np.select(cond, [1, -1], default = 0)

df["pred_signal"] = df["pred_signal"].shift(1) # Since we use today's prediction to decide to trade tommorow

df["z_score"] = (mavg_20d - mavg_60d)/std_60d


#  Generating Trading signals using z_score
df["z_signal"] = 0
df.loc[df["z_score"] < -1, "z_signal"] = 1
df.loc[df["z_score"] > 1, "z_signal"] = -1


# Stop-loss rule
stop_loss_threshold = -0.02
df["stop_loss"] = 0

df["position"] = df["pred_signal"].replace(0, np.nan).ffill().fillna(0)


df["position_change"] = df["position"].diff()
df["entry_price"] = np.nan
df.loc[(df["position_change"] != 0) & (df["position"] != 0), "entry_price"] = df["price"]
df["entry_price"] = df["entry_price"].ffill()
df["trade_ret"] = (df["position"] * (df["price"] - df["entry_price"])/df["entry_price"])
posn = (df["position"] != 0) & (df["position_change"] == 0) & (df["trade_ret"] < stop_loss_threshold)
df.loc[posn, "stop_loss"] = 1
print("Days on which stop_loss is activated:", df[posn].index)


df["ret"] = df["price"].pct_change()

# Maximum Exposure rule
max_exposure = 0.6
df["exposure_position"] = max_exposure * df["position"]
df["strategy_ret_ME"] = df["exposure_position"] * df["ret"]


# Part 6
transaction_cost_rate = 0.001 # 0.1% per trade


df["gross_strategy_ret"] = df["position"] * df["ret"]
df["transaction_costs"] = (df["position_change"].abs()) * transaction_cost_rate
df["strategy_ret"] = df["gross_strategy_ret"] - df["transaction_costs"].fillna(0)
df["cum_ret"] = (1 + df["strategy_ret"]).cumprod() - 1



# Part 7

def get_metrics(df_slice, x):
    ret_series = df_slice["strategy_ret"] if x == 0 else df_slice["ret"]
    sharpe = (ret_series.mean()/ret_series.std()) * np.sqrt(252)
    
    curr_cum = (1 + ret_series).cumprod()
    curr_max = curr_cum.cummax()
    drawdown = (curr_cum/curr_max) - 1
    max_dd = drawdown.min()
    
    total_ret = curr_cum.iloc[-1] - 1

    active_days = ret_series[ret_series != 0]
    if len(active_days) > 0:
        wlr = (active_days > 0).sum() / len(active_days)
    else:
        wlr = 0.0
    
    return sharpe, max_dd, total_ret, wlr

# Calculating metrics for Train(In-Sample) and Test(Out-of-Sample)
train_metrics = get_metrics(df.loc[train_idx], 0)
test_metrics  = get_metrics(df.loc[test_idx], 0)

# Buy-and-Hold Benchmark (bah)
bah_metrics = get_metrics(df.loc[test_idx], 1)
df["cum_bah"] = (1 + df["ret"]).cumprod() - 1

# Comparison DataFrame
Comp = pd.DataFrame(
    {
        "In-Sample (Train)": train_metrics,
        "Out-of-Sample (Test)": test_metrics,
        "Benchmark (Test Only)": bah_metrics
    },
    index=["Sharpe Ratio", "Max Drawdown", "Total Return", "Win-Loss Ratio"]
)

print(Comp)


# THE PLOTS

sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (14, 10)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)


# Kalman Parameters (Beta)
ax1.plot(df.index, df["kalman_beta"], label="Kalman Beta (Slope)", color="purple", alpha=0.8)
ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax1.set_title("Kalman Filter Estimated State (Beta/Momentum)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Beta Value")
ax1.legend()

# Trading Signals
subset = df[:] 
ax2.plot(subset.index, subset["price"], label="MSFT Price", color="black", alpha=0.5)


# Plotting Buy Signals with Green Triangles
buys = subset[subset["pred_signal"] == 1]
ax2.scatter(buys.index, buys["price"], marker="^", color="green", label="Buy Signal", s=100, zorder=5)

# Plotting Sell Signals with Red Triangles
sells = subset[subset["pred_signal"] == -1]
ax2.scatter(sells.index, sells["price"], marker="v", color="red", label="Sell Signal", s=100, zorder=5)

ax2.set_title("Trading Signals", fontsize=12, fontweight="bold")
ax2.set_ylabel("Price ($)")
ax2.legend()


# Equity Curve
ax3.plot(df.index, (1 + df["cum_ret"]) * 100, label="Kalman ML Strategy", color="blue", linewidth=1.5)
ax3.plot(df.index, (1 + df["cum_bah"]) * 100, label="Buy & Hold Benchmark", color="gray", alpha=0.6, linestyle="--")
ax3.set_title("Cumulative Returns Comparison (%)", fontsize=12, fontweight="bold")
ax3.set_ylabel("Return (%)")
ax3.legend()



plt.tight_layout()
plt.savefig("Plots.png")
plt.show()
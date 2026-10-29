"""
Topic 13: Time-series analysis
Libraries: pandas, statsmodels (numpy, matplotlib for support)

Outputs: time-series plot, trend, seasonality, autocorrelation and a basic
forecast, saved to the output folder.

IMPORTANT ABOUT THE DATA
The Students Performance dataset has no date or time column, so it cannot be
analysed as a time series directly. This script therefore builds a SIMULATED
monthly series of the average math score of a student cohort. Its level and
spread come from the real dataset, and a trend, a yearly seasonal pattern and
random noise are added with known values. State in your report that the series
is simulated. Because the true trend and seasonality are known, we can check
that the analysis recovers them.

To use a real series instead, set SOURCE = "co2" (a monthly CO2 series that is
bundled with statsmodels, so nothing has to be downloaded).
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning)   # hides a statsmodels notice

SOURCE = "student_simulated"      # or "co2"
PERIOD = 12                       # monthly data, yearly seasonality
HORIZON = 12                      # months to forecast

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


# ---------- Build or load the series ----------
def build_student_series():
    df = pd.read_csv(HERE.parent / "data" / "Cleaned_Students_Performance.csv")
    df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]
    col = next(c for c in df.columns if "math" in c)
    scores = df[col].dropna().to_numpy(dtype=float)
    mu, sigma = scores.mean(), scores.std(ddof=1)

    rng = np.random.default_rng(42)
    idx = pd.date_range("2019-01-01", periods=72, freq="MS")      # 6 years
    t = np.arange(len(idx))
    trend = 0.08 * t                                               # +0.08 points per month
    season = 3.0 * np.sin(2 * np.pi * (t % 12) / 12)               # yearly cycle
    noise = rng.normal(0, sigma / np.sqrt(40), len(idx))           # mean of ~40 students
    series = pd.Series(mu - 0.08 * 35 + trend + season + noise, index=idx,
                       name="Average math score (simulated)")
    return series, "Average math score (simulated cohort)", "Score"


def load_co2_series():
    co2 = sm.datasets.co2.load_pandas().data["co2"]
    series = co2.resample("MS").mean().interpolate().loc["1990":"1999"]
    series.name = "CO2"
    return series, "Atmospheric CO2 (monthly mean)", "ppm"


series, title, unit = build_student_series() if SOURCE == "student_simulated" else load_co2_series()
series.to_csv(OUT / "time_series_data.csv", header=True)
print(f"\nSeries: {title}")
print(f"{len(series)} monthly observations from {series.index[0]:%b %Y} to {series.index[-1]:%b %Y}")
print(series.describe().round(3))

# ---------- Trend: moving average and linear trend ----------
t_num = np.arange(len(series))
slope, intercept = np.polyfit(t_num, series.to_numpy(), 1)
linear_trend = pd.Series(intercept + slope * t_num, index=series.index)
moving_avg = series.rolling(window=PERIOD, center=True).mean()
print(f"\nTREND: linear trend slope = {slope:.4f} {unit} per month "
      f"({slope * 12:.3f} per year)")
if SOURCE == "student_simulated":
    print("       (true slope used in the simulation: 0.0800 per month; the estimate differs a little because of the random noise)")

# ---------- Decomposition: trend + seasonality + residual ----------
decomp = seasonal_decompose(series, model="additive", period=PERIOD)
seasonal_pattern = decomp.seasonal.iloc[:PERIOD]
seasonal_pattern.index = seasonal_pattern.index.strftime("%b")
print("\nSEASONALITY: average effect of each month (additive)\n")
print(seasonal_pattern.round(3).to_string())
peak, trough = seasonal_pattern.idxmax(), seasonal_pattern.idxmin()
print(f"\nPeak month: {peak} ({seasonal_pattern.max():+.2f}); "
      f"lowest month: {trough} ({seasonal_pattern.min():+.2f})")
pd.DataFrame({"observed": decomp.observed, "trend": decomp.trend,
              "seasonal": decomp.seasonal, "residual": decomp.resid}
             ).round(4).to_csv(OUT / "decomposition.csv")

# ---------- Stationarity and autocorrelation ----------
adf_level = adfuller(series.to_numpy())
diff_series = series.diff(PERIOD).dropna()                 # seasonal differencing
adf_diff = adfuller(diff_series.to_numpy())
print("\nAUGMENTED DICKEY-FULLER TEST (H0: the series has a unit root / is non-stationary)")
print(f"  original series        : statistic = {adf_level[0]:.3f}, p = {adf_level[1]:.4f}")
print(f"  after seasonal differencing: statistic = {adf_diff[0]:.3f}, p = {adf_diff[1]:.4f}")

acf_vals = sm.tsa.acf(series, nlags=PERIOD, fft=False)
print("\nAUTOCORRELATION at lags 1 to 12:", np.round(acf_vals[1:], 3))
print(f"  Lag 1 autocorrelation = {acf_vals[1]:.3f}; "
      f"lag 12 autocorrelation = {acf_vals[PERIOD]:.3f}")

# ---------- Forecasting: Holt-Winters with a hold-out test ----------
train, test = series.iloc[:-HORIZON], series.iloc[-HORIZON:]


def holt_winters(data):
    return ExponentialSmoothing(data, trend="add", seasonal="add",
                                seasonal_periods=PERIOD,
                                initialization_method="estimated").fit()


hw_test = holt_winters(train)
pred_test = hw_test.forecast(HORIZON)
naive_test = pd.Series(train.iloc[-PERIOD:].to_numpy()[:HORIZON], index=test.index)   # seasonal naive


def metrics(actual, pred):
    err = actual.to_numpy() - pred.to_numpy()
    return np.mean(np.abs(err)), np.sqrt(np.mean(err**2))


mae_hw, rmse_hw = metrics(test, pred_test)
mae_nv, rmse_nv = metrics(test, naive_test)
accuracy = pd.DataFrame({"model": ["Holt-Winters", "Seasonal naive (repeat last year)"],
                         "MAE": [mae_hw, mae_nv], "RMSE": [rmse_hw, rmse_nv]}).round(4)
accuracy.to_csv(OUT / "forecast_accuracy.csv", index=False)
print(f"\nFORECAST CHECK on the last {HORIZON} months (model fitted on the earlier data)\n")
print(accuracy.to_string(index=False))

# Final forecast: refit on all data, forecast the next HORIZON months
hw_full = holt_winters(series)
future_idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")
forecast = pd.Series(hw_full.forecast(HORIZON).to_numpy(), index=future_idx)
resid_sd = np.std(hw_full.resid, ddof=1)
lower, upper = forecast - 1.96 * resid_sd, forecast + 1.96 * resid_sd
fc_table = pd.DataFrame({"forecast": forecast, "lower_95": lower, "upper_95": upper}).round(3)
fc_table.to_csv(OUT / "forecast_next_12_months.csv")
print(f"\nFORECAST for the next {HORIZON} months (approximate 95% interval)\n")
print(fc_table.to_string())

# ---------- Interpretation ----------
print("\nInterpretation")
print(f"- Trend: the series moves {'up' if slope > 0 else 'down'} by about "
      f"{abs(slope) * 12:.2f} {unit} per year.")
print(f"- Seasonality: a repeating yearly pattern with its high in {peak} and low in {trough}.")
print(f"- The original series is {'non-stationary' if adf_level[1] > 0.05 else 'stationary'} "
      f"(ADF p = {adf_level[1]:.3f}); seasonal differencing "
      f"{'makes it stationary' if adf_diff[1] < 0.05 else 'does not fully remove the non-stationarity'} "
      f"(p = {adf_diff[1]:.3f}).")
print(f"- Autocorrelation is {acf_vals[1]:.2f} at lag 1 and {acf_vals[PERIOD]:.2f} at lag 12: "
      "this month's value is related to recent months and to the same month last year.")
print(f"- Holt-Winters {'beats' if rmse_hw < rmse_nv else 'does not beat'} the seasonal naive baseline "
      f"on the hold-out period (RMSE {rmse_hw:.2f} vs {rmse_nv:.2f}).")
print("- Forecast intervals widen in practice with the horizon; the interval shown "
      "is a simple approximation based on the model residuals.")

# ---------- Charts ----------
# Figure 1: time-series plot with trend
fig1, ax = plt.subplots(figsize=(12, 5))
ax.plot(series.index, series, color="steelblue", lw=1.5, label="Series")
ax.plot(moving_avg.index, moving_avg, color="orange", lw=2, label="12-month moving average")
ax.plot(linear_trend.index, linear_trend, color="red", linestyle="--", lw=2,
        label=f"Linear trend ({slope:.3f} per month)")
ax.set_title(title)
ax.set_xlabel("Date")
ax.set_ylabel(unit)
ax.legend()
fig1.tight_layout()
fig1.savefig(OUT / "time_series_plot.png", dpi=150)

# Figure 2: decomposition
fig2, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
for ax, comp, name in zip(axes, [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
                          ["Observed", "Trend", "Seasonal", "Residual"]):
    ax.plot(comp, color="navy" if name != "Residual" else "grey",
            marker="." if name == "Residual" else None, lw=1.2 if name != "Residual" else 0)
    ax.set_ylabel(name)
axes[0].set_title("Decomposition into trend, seasonality and residual")
fig2.tight_layout()
fig2.savefig(OUT / "decomposition.png", dpi=150)

# Figure 3: autocorrelation
fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
plot_acf(series, lags=24, ax=ax1)
ax1.set_title("Autocorrelation (ACF)")
plot_pacf(series, lags=24, ax=ax2, method="ywm")
ax2.set_title("Partial autocorrelation (PACF)")
plot_acf(diff_series, lags=18, ax=ax3)
ax3.set_title("ACF after seasonal differencing")
fig3.tight_layout()
fig3.savefig(OUT / "autocorrelation_plots.png", dpi=150)

# Figure 4: forecast
fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train.index, train, color="steelblue", label="Training data")
ax1.plot(test.index, test, color="black", label="Actual (hold-out)")
ax1.plot(pred_test.index, pred_test, color="red", linestyle="--", label="Holt-Winters forecast")
ax1.plot(naive_test.index, naive_test, color="green", linestyle=":", label="Seasonal naive")
ax1.set_title("Forecast check on the last 12 months")
ax1.legend(fontsize=8)

ax2.plot(series.index, series, color="steelblue", label="Observed")
ax2.plot(forecast.index, forecast, color="red", lw=2, label="Forecast")
ax2.fill_between(forecast.index, lower, upper, color="red", alpha=0.2, label="Approx. 95% interval")
ax2.set_title(f"Forecast for the next {HORIZON} months")
ax2.legend(fontsize=8)
for ax in (ax1, ax2):
    ax.set_xlabel("Date")
    ax.set_ylabel(unit)
fig4.tight_layout()
fig4.savefig(OUT / "forecast_plot.png", dpi=150)

print(f"\nSaved results and charts to: {OUT}")
plt.show()
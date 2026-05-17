"""
Simple Pairs Trading Strategy Using Cointegration
==================================================
S&P 500 stocks | Engle-Granger cointegration | Mean-reversion backtest
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy import stats
from itertools import combinations
import os

# ── Configuration ────────────────────────────────────────────────────────────
INSAMPLE_START   = "2018-01-01"
INSAMPLE_END     = "2021-12-31"
OUTSAMPLE_START  = "2022-01-01"
OUTSAMPLE_END    = "2023-12-31"

COINT_PVAL       = 0.05        # cointegration p-value threshold
ENTRY_ZSCORE     = 2.0         # open trade when |z| > this
EXIT_ZSCORE      = 0.0         # close trade when |z| < this
STOP_LOSS_ZSCORE = 3.5         # stop-loss
MAX_HOLD_DAYS    = 60          # maximum holding period
TRANSACTION_COST = 0.001       # 10 bps each side
LOOKBACK         = 60          # rolling window for hedge ratio & z-score

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Data Download ──────────────────────────────────────────────────────────

def get_sp500_tickers(n: int = 50) -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia; return first n for speed."""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"  Fetched {len(tickers)} tickers from Wikipedia.")
        return tickers[:n]
    except Exception as e:
        print(f"  Wikipedia fetch failed ({e}). Using fallback list.")
        return [
            "AAPL","MSFT","GOOGL","AMZN","META","NVDA","BRK-B","JPM","V","UNH",
            "XOM","JNJ","PG","MA","HD","CVX","MRK","ABBV","PEP","KO",
            "LLY","AVGO","COST","WMT","BAC","MCD","TMO","ACN","CSCO","ABT",
            "CRM","DHR","TXN","NEE","WFC","VZ","PM","RTX","HON","AMGN",
            "BMY","ORCL","QCOM","IBM","GE","CAT","LOW","SPGI","GS","BLK"
        ]


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices; drop columns with > 10% missing."""
    print(f"\n[1] Downloading prices for {len(tickers)} tickers …")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(axis=1, thresh=int(0.90 * len(raw)))
    raw = raw.ffill().dropna()
    print(f"    Retained {raw.shape[1]} stocks, {raw.shape[0]} trading days.")
    return raw

# ── 2. Cointegration Screening ────────────────────────────────────────────────

def test_cointegration(prices: pd.DataFrame, pval_thresh: float = COINT_PVAL):
    """
    Run Engle-Granger cointegration test on all pairs.
    Returns a DataFrame of significant pairs sorted by p-value.
    """
    tickers = prices.columns.tolist()
    n = len(tickers)
    results = []

    print(f"\n[2] Testing cointegration for {n*(n-1)//2} pairs …")
    for s1, s2 in combinations(tickers, 2):
        score, pval, _ = coint(prices[s1], prices[s2])
        if pval < pval_thresh:
            corr = prices[s1].corr(prices[s2])
            results.append({"stock1": s1, "stock2": s2, "pvalue": pval,
                            "coint_score": score, "correlation": corr})

    df = pd.DataFrame(results).sort_values("pvalue").reset_index(drop=True)
    print(f"    Found {len(df)} cointegrated pairs (p < {pval_thresh}).")
    return df


def plot_pvalue_heatmap(prices: pd.DataFrame, pairs_df: pd.DataFrame, fname: str):
    tickers = prices.columns.tolist()
    mat = pd.DataFrame(np.nan, index=tickers, columns=tickers)
    for _, row in pairs_df.iterrows():
        mat.loc[row.stock1, row.stock2] = row.pvalue
        mat.loc[row.stock2, row.stock1] = row.pvalue

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(mat.fillna(1), cmap="RdYlGn_r", vmin=0, vmax=0.1,
                ax=ax, cbar_kws={"label": "Cointegration p-value"})
    ax.set_title("Cointegration p-value Heatmap (green = more significant)")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"    Saved: {fname}")

# ── 3. Spread & Z-score ───────────────────────────────────────────────────────

def compute_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """OLS hedge ratio: regress y on x."""
    model = OLS(y, add_constant(x)).fit()
    return model.params.iloc[1]


def compute_spread_zscore(y: pd.Series, x: pd.Series,
                          lookback: int = LOOKBACK) -> pd.DataFrame:
    """Rolling OLS hedge ratio → spread → rolling z-score."""
    out = pd.DataFrame(index=y.index)
    hedge = np.full(len(y), np.nan)
    spread = np.full(len(y), np.nan)

    for i in range(lookback, len(y)):
        window_y = y.iloc[i - lookback: i]
        window_x = x.iloc[i - lookback: i]
        h = compute_hedge_ratio(window_y, window_x)
        hedge[i] = h
        spread[i] = y.iloc[i] - h * x.iloc[i]

    out["hedge"] = hedge
    out["spread"] = spread
    out["spread_mean"] = pd.Series(spread, index=y.index).rolling(lookback).mean()
    out["spread_std"]  = pd.Series(spread, index=y.index).rolling(lookback).std()
    out["zscore"] = (out["spread"] - out["spread_mean"]) / out["spread_std"]
    return out.dropna()


def plot_spread_zscore(signal: pd.DataFrame, s1: str, s2: str,
                       entry: float, exit_: float, fname: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(signal["spread"], color="steelblue", linewidth=0.8)
    axes[0].axhline(signal["spread_mean"].iloc[-1], color="k", linestyle="--", linewidth=0.7)
    axes[0].set_title(f"Spread: {s1} − β·{s2}")
    axes[0].set_ylabel("Spread")

    axes[1].plot(signal["zscore"], color="purple", linewidth=0.8)
    axes[1].axhline(entry,   color="red",   linestyle="--", linewidth=0.9, label=f"Entry ±{entry}")
    axes[1].axhline(-entry,  color="red",   linestyle="--", linewidth=0.9)
    axes[1].axhline(exit_,   color="green", linestyle="--", linewidth=0.9, label=f"Exit ±{exit_}")
    axes[1].axhline(-exit_,  color="green", linestyle="--", linewidth=0.9)
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_title("Z-score")
    axes[1].set_ylabel("Z-score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# ── 4. Backtester ─────────────────────────────────────────────────────────────

def backtest_pair(signal: pd.DataFrame,
                  entry: float        = ENTRY_ZSCORE,
                  exit_: float        = EXIT_ZSCORE,
                  stop_loss: float    = STOP_LOSS_ZSCORE,
                  max_hold: int       = MAX_HOLD_DAYS,
                  tc: float           = TRANSACTION_COST) -> pd.DataFrame:
    """
    Long spread when z < -entry; short spread when z > +entry.
    Close when |z| < exit_ or stop-loss / max-hold hit.
    Returns daily P&L series and trade log.
    """
    zscores = signal["zscore"].values
    dates   = signal.index
    n       = len(zscores)

    position      = 0   # +1 long spread, -1 short spread, 0 flat
    entry_price   = 0.0
    hold_days     = 0
    pnl           = np.zeros(n)
    trades        = []

    prev_spread = signal["spread"].values.copy()

    for i in range(1, n):
        z = zscores[i]
        spread_ret = prev_spread[i] - prev_spread[i - 1]

        if position != 0:
            hold_days += 1
            pnl[i] = position * spread_ret

            # Exit conditions
            close = False
            reason = ""
            if position == 1  and z > -exit_:  close, reason = True, "mean-revert"
            if position == -1 and z <  exit_:  close, reason = True, "mean-revert"
            if abs(z) > stop_loss:             close, reason = True, "stop-loss"
            if hold_days >= max_hold:           close, reason = True, "max-hold"

            if close:
                pnl[i] -= tc  # exit cost
                trade_ret = sum(pnl[max(0, i - hold_days): i + 1])
                trades.append({"date": dates[i], "position": position,
                               "hold_days": hold_days, "return": trade_ret, "reason": reason})
                position, hold_days = 0, 0

        else:
            if z < -entry:
                position  = 1
                hold_days = 0
                pnl[i]   -= tc  # entry cost
            elif z > entry:
                position  = -1
                hold_days = 0
                pnl[i]   -= tc

    result = pd.DataFrame({"date": dates, "pnl": pnl}).set_index("date")
    result["cumret"] = result["pnl"].cumsum()
    trade_log = pd.DataFrame(trades)
    return result, trade_log


def performance_metrics(pnl_series: pd.Series, trade_log: pd.DataFrame, label: str = "") -> dict:
    daily = pnl_series.values
    cumret = np.cumsum(daily)
    ann_ret = daily.mean() * 252
    ann_vol = daily.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0

    roll_max = np.maximum.accumulate(cumret)
    drawdown = cumret - roll_max
    max_dd   = drawdown.min()

    win_rate    = (trade_log["return"] > 0).mean() if len(trade_log) else np.nan
    avg_ret     = trade_log["return"].mean()        if len(trade_log) else np.nan
    n_trades    = len(trade_log)

    return {
        "label":       label,
        "ann_return":  round(ann_ret,  4),
        "ann_vol":     round(ann_vol,  4),
        "sharpe":      round(sharpe,   4),
        "max_drawdown":round(max_dd,   4),
        "n_trades":    n_trades,
        "win_rate":    round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
        "avg_trade":   round(avg_ret,  4) if not np.isnan(avg_ret)  else np.nan,
    }

# ── 5. Parameter Optimisation ─────────────────────────────────────────────────

def optimize_parameters(signal: pd.DataFrame, s1: str, s2: str) -> pd.DataFrame:
    """Grid-search over entry/exit thresholds on in-sample data."""
    entry_range = [1.5, 2.0, 2.5]
    exit_range  = [0.0, 0.25, 0.5]
    rows = []
    for ent in entry_range:
        for ex in exit_range:
            res, tlog = backtest_pair(signal, entry=ent, exit_=ex)
            m = performance_metrics(res["pnl"], tlog, label=f"entry={ent}, exit={ex}")
            m["entry"] = ent
            m["exit"]  = ex
            rows.append(m)
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print(f"\n    Parameter grid for {s1}-{s2}:")
    print(df[["entry","exit","sharpe","ann_return","max_drawdown","n_trades"]].to_string(index=False))
    return df


def plot_sensitivity(opt_df: pd.DataFrame, s1: str, s2: str, fname: str):
    pivot = opt_df.pivot(index="entry", columns="exit", values="sharpe")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
    ax.set_title(f"Sharpe Ratio Sensitivity — {s1}/{s2}")
    ax.set_xlabel("Exit z-score")
    ax.set_ylabel("Entry z-score")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# ── 6. Portfolio Equity Curve ─────────────────────────────────────────────────

def plot_equity_curve(pnl_list: list[pd.Series], labels: list[str],
                      title: str, fname: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    portfolio = pd.concat(pnl_list, axis=1).fillna(0).sum(axis=1).cumsum()
    for pnl, lbl in zip(pnl_list, labels):
        ax.plot(pnl.cumsum(), linewidth=0.7, alpha=0.5, label=lbl)
    ax.plot(portfolio, linewidth=2, color="black", label="Portfolio")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("Cumulative P&L (spread units)")
    ax.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# ── 7. ADF Verification ───────────────────────────────────────────────────────

def adf_test(series: pd.Series, label: str = "") -> None:
    result = adfuller(series.dropna())
    print(f"    ADF [{label}]: stat={result[0]:.3f}, p={result[1]:.4f} "
          f"({'stationary' if result[1] < 0.05 else 'non-stationary'})")

# ── 8. Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PAIRS TRADING — COINTEGRATION STRATEGY")
    print("=" * 60)

    # --- Data ---
    tickers = get_sp500_tickers(n=50)
    all_prices = download_prices(tickers,
                                 start=INSAMPLE_START, end=OUTSAMPLE_END)

    insample   = all_prices.loc[INSAMPLE_START:INSAMPLE_END]
    outsample  = all_prices.loc[OUTSAMPLE_START:OUTSAMPLE_END]

    # --- Find cointegrated pairs (in-sample) ---
    pairs_df = test_cointegration(insample, pval_thresh=COINT_PVAL)
    if pairs_df.empty:
        print("  No cointegrated pairs found. Try relaxing COINT_PVAL.")
        return

    pairs_df.to_csv(os.path.join(OUTPUT_DIR, "cointegration_pvalues.csv"), index=False)
    plot_pvalue_heatmap(insample, pairs_df,
                        os.path.join(OUTPUT_DIR, "coint_heatmap.png"))

    # --- Select top pairs (max 5) ---
    top_pairs = pairs_df.head(5)
    print(f"\n[3] Top {len(top_pairs)} cointegrated pairs:")
    print(top_pairs[["stock1","stock2","pvalue","correlation"]].to_string(index=False))

    # --- Build signals, optimise, and backtest ---
    is_pnl_list, oos_pnl_list, labels = [], [], []
    perf_rows = []

    for _, row in top_pairs.iterrows():
        s1, s2 = row["stock1"], row["stock2"]
        label  = f"{s1}/{s2}"
        print(f"\n{'─'*50}")
        print(f"  Pair: {label}")

        # In-sample signal
        is_sig = compute_spread_zscore(insample[s1], insample[s2], LOOKBACK)
        adf_test(is_sig["spread"], f"{label} IS spread")

        # Plot spread/z-score
        plot_spread_zscore(is_sig, s1, s2, ENTRY_ZSCORE, EXIT_ZSCORE,
                           os.path.join(OUTPUT_DIR, f"spread_{s1}_{s2}.png"))

        # Parameter optimisation
        opt_df = optimize_parameters(is_sig, s1, s2)
        plot_sensitivity(opt_df, s1, s2,
                         os.path.join(OUTPUT_DIR, f"sensitivity_{s1}_{s2}.png"))

        # Best params from optimisation
        best = opt_df.iloc[0]
        best_entry, best_exit = best["entry"], best["exit"]

        # In-sample backtest
        is_res, is_trades = backtest_pair(is_sig, entry=best_entry, exit_=best_exit)
        is_metrics = performance_metrics(is_res["pnl"], is_trades, f"{label} IS")
        is_pnl_list.append(is_res["pnl"])

        # Out-of-sample signal
        oos_sig = compute_spread_zscore(outsample[s1], outsample[s2], LOOKBACK)
        oos_res, oos_trades = backtest_pair(oos_sig, entry=best_entry, exit_=best_exit)
        oos_metrics = performance_metrics(oos_res["pnl"], oos_trades, f"{label} OOS")
        oos_pnl_list.append(oos_res["pnl"])

        labels.append(label)

        print(f"\n    In-Sample  : Sharpe={is_metrics['sharpe']:.2f}, "
              f"AnnRet={is_metrics['ann_return']:.2%}, "
              f"MaxDD={is_metrics['max_drawdown']:.4f}, "
              f"Trades={is_metrics['n_trades']}")
        print(f"    Out-Sample : Sharpe={oos_metrics['sharpe']:.2f}, "
              f"AnnRet={oos_metrics['ann_return']:.2%}, "
              f"MaxDD={oos_metrics['max_drawdown']:.4f}, "
              f"Trades={oos_metrics['n_trades']}")

        # Trade log
        if not is_trades.empty:
            is_trades.to_csv(
                os.path.join(OUTPUT_DIR, f"trade_log_{s1}_{s2}_IS.csv"), index=False)

        perf_rows += [is_metrics, oos_metrics]

    # --- Portfolio equity curves ---
    if is_pnl_list:
        plot_equity_curve(is_pnl_list, labels,
                          "In-Sample Portfolio Equity Curve",
                          os.path.join(OUTPUT_DIR, "equity_curve_IS.png"))
    if oos_pnl_list:
        plot_equity_curve(oos_pnl_list, labels,
                          "Out-of-Sample Portfolio Equity Curve",
                          os.path.join(OUTPUT_DIR, "equity_curve_OOS.png"))

    # --- Summary table ---
    summary = pd.DataFrame(perf_rows)
    print(f"\n{'='*60}")
    print("  PERFORMANCE SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, "performance_summary.csv"), index=False)

    print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()

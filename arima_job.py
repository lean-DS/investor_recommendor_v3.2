# arima_job.py
# Batch ARIMA forecaster for SP500 / FTSE100 / FTSE250
# Reads and writes directly to GCS (requires gcsfs installed).

import os
import sys
import json
import time
import math
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict, Optional

from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import BDay

# ---------- Config via env ----------
BUCKET = os.environ.get("GCS_DATA_BUCKET", "fintech-inv-recomm-portfolio-data")
BASE   = os.environ.get("GCS_BASE_PREFIX", "portfolio_data")
# Comma-separated list: "sp500,ftse100,ftse250"
INDEXES = [s.strip().lower() for s in os.environ.get("JOB_INDEXES", "sp500,ftse100,ftse250").split(",") if s.strip()]
# Forecast horizon in trading days (63 ≈ 3M, 126 ≈ 6M)
H = int(os.environ.get("JOB_HORIZON_DAYS", "126"))
# Optional cap of tickers per index (0 = all)
CAP = int(os.environ.get("JOB_CAP_TICKERS", "0"))

# Output layout in GCS
RUN_STAMP = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
LATEST_PREFIX = f"gs://{BUCKET}/{BASE}/ml/arima_statsmodels/latest"
RUNS_PREFIX   = f"gs://{BUCKET}/{BASE}/ml/arima_statsmodels/runs/{RUN_STAMP}"

# File names inside each index folder
PRICE_FILES = {
    "sp500":  f"gs://{BUCKET}/{BASE}/sp500/sp500_prices.parquet",
    "ftse100":f"gs://{BUCKET}/{BASE}/ftse100/ftse100_prices.parquet",
    "ftse250":f"gs://{BUCKET}/{BASE}/ftse250/ftse250_prices.parquet",
}

# ---------- Small utilities ----------
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0

def ensure_trading_dates(start_dt: pd.Timestamp, h: int) -> List[pd.Timestamp]:
    dates = []
    d = start_dt
    for _ in range(h):
        d = d + BDay(1)
        dates.append(pd.Timestamp(d).normalize())
    return dates

def best_arima_order(y: pd.Series) -> Optional[Tuple[int,int,int]]:
    """
    Tiny ARIMA order search for speed. Returns order with lowest AIC.
    Skips if not enough data.
    """
    candidates = [
        (1,1,0), (2,1,0), (1,1,1), (2,1,1), (5,1,0), (2,1,2)
    ]
    y = y.dropna().astype(float)
    if len(y) < 40:
        return None

    best = (None, np.inf)
    for order in candidates:
        try:
            m = ARIMA(y, order=order)
            r = m.fit(method_kwargs={"warn_convergence": False})
            aic = getattr(r, "aic", np.inf)
            if aic < best[1]:
                best = (order, aic)
        except Exception:
            continue
    return best[0]

def fit_forecast_one(y: pd.Series, horizon: int) -> Dict:
    """
    Train/test split for quick metrics + future forecast.
    Returns dict with metrics and future predictions.
    """
    y = y.dropna().astype(float)
    if len(y) < 60:
        raise ValueError("Not enough history")

    # quick backtest on last 60 points
    test_n = min(60, int(len(y) * 0.2))  # ~20% or max 60
    train = y.iloc[:-test_n]
    test  = y.iloc[-test_n:]

    order = best_arima_order(train)
    if order is None:
        raise ValueError("No valid ARIMA order found")

    # Fit on train for metrics
    m = ARIMA(train, order=order)
    r = m.fit(method_kwargs={"warn_convergence": False})

    # 1-step rolling predict over test
    preds = []
    hist = train.copy()
    for t in test:
        try:
            rr = ARIMA(hist, order=order).fit(method_kwargs={"warn_convergence": False})
            p = rr.forecast(1)
            preds.append(float(p.iloc[0]))
            hist = pd.concat([hist, pd.Series([t], index=[hist.index[-1] + BDay(1)])])
        except Exception:
            preds.append(np.nan)

    # Metrics
    valid_mask = ~np.isnan(preds)
    if valid_mask.sum() == 0:
        rmse = mae = mape = smape_v = np.nan
    else:
        y_true = test.values[valid_mask]
        y_pred = np.array(preds, dtype=float)[valid_mask]
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        mape_v = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1, y_true))) * 100.0)
        smape_v = float(smape(y_true, y_pred))

    # Refit on full data for future forecast
    m_full = ARIMA(y, order=order)
    r_full = m_full.fit(method_kwargs={"warn_convergence": False})
    f = r_full.get_forecast(steps=horizon)
    fc = f.predicted_mean
    ci = f.conf_int(alpha=0.05)
    lower = ci.iloc[:, 0].rename("yhat_lower")
    upper = ci.iloc[:, 1].rename("yhat_upper")

    # build dates
    last_dt = y.index.max()
    if not isinstance(last_dt, pd.Timestamp):
        last_dt = pd.to_datetime(last_dt)
    future_idx = ensure_trading_dates(last_dt, horizon)

    preds_df = pd.DataFrame({
        "Date": future_idx,
        "yhat": fc.values[:horizon],
        "yhat_lower": lower.values[:horizon],
        "yhat_upper": upper.values[:horizon],
    })

    return {
        "order": order,
        "metrics": {"RMSE": rmse, "MAE": mae, "MAPE": mape_v, "sMAPE": smape_v, "TestPoints": int(test_n)},
        "preds": preds_df
    }

def read_prices(index_name: str) -> pd.DataFrame:
    """Return Close prices pivoted: Date x Ticker"""
    fpath = PRICE_FILES[index_name]
    df = pd.read_parquet(fpath)
    # Expect columns: Date, Close, Ticker
    df["Date"] = pd.to_datetime(df["Date"])
    px = (df[["Date","Ticker","Close"]]
          .pivot(index="Date", columns="Ticker", values="Close")
          .sort_index())
    return px

def write_gcs(df: pd.DataFrame, gcs_uri: str):
    # pandas + gcsfs handles gs:// URIs directly
    ext = gcs_uri.split(".")[-1].lower()
    if ext == "parquet":
        df.to_parquet(gcs_uri, index=False)
    elif ext in ("csv", "txt"):
        df.to_csv(gcs_uri, index=False)
    else:
        # default parquet
        df.to_parquet(gcs_uri, index=False)

def run_index(index_name: str):
    print(f"\n=== Running ARIMA for index: {index_name} ===")
    px = read_prices(index_name)

    # cap columns if requested
    tickers = list(px.columns)
    if CAP and CAP > 0:
        tickers = tickers[:CAP]

    all_metrics = []
    all_preds   = []

    for i, t in enumerate(tickers, 1):
        y = px[t].dropna()
        y.index = pd.to_datetime(y.index)
        try:
            res = fit_forecast_one(y, H)
            m = res["metrics"].copy()
            m.update({"Ticker": t, "Order": str(res["order"]), "H": H, "Index": index_name})
            dfp = res["preds"].copy()
            dfp.insert(1, "Ticker", t)
            all_metrics.append(m)
            all_preds.append(dfp)
            print(f"  [{i:>4}/{len(tickers)}] {t}: RMSE={m['RMSE']:.3f} sMAPE={m['sMAPE']:.2f}%")
        except Exception as e:
            print(f"  [{i:>4}/{len(tickers)}] {t}: ERROR -> {e}")
            # continue

    if not all_preds:
        print("No predictions produced for this index.")
        return

    preds_df   = pd.concat(all_preds, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)

    # Write to LATEST and timestamped RUNS
    latest_pred_uri   = f"{LATEST_PREFIX}/{index_name}/preds.parquet"
    latest_metric_uri = f"{LATEST_PREFIX}/{index_name}/metrics.csv"
    run_pred_uri      = f"{RUNS_PREFIX}/{index_name}/preds.parquet"
    run_metric_uri    = f"{RUNS_PREFIX}/{index_name}/metrics.csv"

    for uri, df in [
        (latest_pred_uri, preds_df),
        (latest_metric_uri, metrics_df),
        (run_pred_uri, preds_df),
        (run_metric_uri, metrics_df),
    ]:
        write_gcs(df, uri)
        print(f"  → wrote {uri}")

def main():
    t0 = time.time()
    print("ARIMA job starting…")
    print(f"BUCKET={BUCKET}, BASE={BASE}, INDEXES={INDEXES}, H={H}, CAP={CAP}")
    errs = {}
    for idx in INDEXES:
        try:
            run_index(idx)
        except Exception as e:
            errs[idx] = str(e)
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nARIMA job finished in {elapsed/60:.1f} min.")
    if errs:
        print("Errors:", json.dumps(errs, indent=2))
        # Non-zero exit so scheduler/job surface failure if *everything* failed
        if len(errs) == len(INDEXES):
            sys.exit(2)

if __name__ == "__main__":
    main()

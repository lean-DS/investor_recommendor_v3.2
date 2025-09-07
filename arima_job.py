# arima_job.py
# Batch ARIMA forecaster for SP500 / FTSE100 / FTSE250
# Reads and writes directly to GCS (requires gcsfs installed).

# arima_job.py
# Batch ARIMA forecaster for SP500 / FTSE100 / FTSE250
# Runs in Cloud Run Jobs with parallel tasks (1 index per task).
# Reads/writes directly to GCS (requires gcsfs installed).

import os
import sys
import json
import time
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
INDEXES = [s.strip().lower() for s in os.environ.get("JOB_INDEXES", "sp500,ftse100,ftse250").split(",") if s.strip()]

# Cloud Run Job fan-out: which index to run
task_idx = os.environ.get("CLOUD_RUN_TASK_INDEX")
if task_idx is not None:
    try:
        task_idx = int(task_idx)
        if task_idx < len(INDEXES):
            INDEXES = [INDEXES[task_idx]]  # just one index per task
        else:
            print(f"Invalid CLOUD_RUN_TASK_INDEX={task_idx}, no such index.")
            sys.exit(1)
    except Exception:
        pass  # fallback → process all

# Forecast horizon in trading days (63 ≈ 3M, 126 ≈ 6M)
H = int(os.environ.get("JOB_HORIZON_DAYS", "126"))
# Optional cap of tickers per index (0 = all)
CAP = int(os.environ.get("JOB_CAP_TICKERS", "0"))

# Output layout in GCS
RUN_STAMP = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
LATEST_PREFIX = f"gs://{BUCKET}/{BASE}/ml/arima_statsmodels/latest"
RUNS_PREFIX   = f"gs://{BUCKET}/{BASE}/ml/arima_statsmodels/runs/{RUN_STAMP}"

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
    candidates = [(1,1,0), (2,1,0), (1,1,1), (2,1,1), (5,1,0), (2,1,2)]
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
    y = y.dropna().astype(float)
    if len(y) < 60:
        raise ValueError("Not enough history")

    test_n = min(60, int(len(y) * 0.2))
    train, test = y.iloc[:-test_n], y.iloc[-test_n:]

    order = best_arima_order(train)
    if order is None:
        raise ValueError("No valid ARIMA order")

    # Fit on train for metrics
    m = ARIMA(train, order=order)
    r = m.fit(method_kwargs={"warn_convergence": False})

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

    valid_mask = ~np.isnan(preds)
    if valid_mask.sum() == 0:
        rmse = mae = mape = smape_v = np.nan
    else:
        y_true = test.values[valid_mask]
        y_pred = np.array(preds, dtype=float)[valid_mask]
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1, y_true))) * 100.0)
        smape_v = float(smape(y_true, y_pred))

    # Refit full
    m_full = ARIMA(y, order=order)
    r_full = m_full.fit(method_kwargs={"warn_convergence": False})
    f = r_full.get_forecast(steps=horizon)
    fc, ci = f.predicted_mean, f.conf_int(alpha=0.05)
    lower, upper = ci.iloc[:,0].rename("yhat_lower"), ci.iloc[:,1].rename("yhat_upper")

    last_dt = y.index.max()
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
        "metrics": {"RMSE": rmse, "MAE": mae, "MAPE": mape, "sMAPE": smape_v, "TestPoints": int(test_n)},
        "preds": preds_df
    }

def read_prices(index_name: str) -> pd.DataFrame:
    fpath = PRICE_FILES[index_name]
    df = pd.read_parquet(fpath)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.pivot(index="Date", columns="Ticker", values="Close").sort_index()

def write_gcs(df: pd.DataFrame, gcs_uri: str):
    ext = gcs_uri.split(".")[-1].lower()
    if ext == "parquet":
        df.to_parquet(gcs_uri, index=False)
    elif ext in ("csv", "txt"):
        df.to_csv(gcs_uri, index=False)
    else:
        df.to_parquet(gcs_uri, index=False)

def run_index(index_name: str):
    print(f"\n=== Running ARIMA for index: {index_name} ===")
    px = read_prices(index_name)

    tickers = list(px.columns)
    if CAP > 0:
        tickers = tickers[:CAP]

    all_metrics, all_preds = [], []
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
            print(f"  [{i}/{len(tickers)}] {t}: RMSE={m['RMSE']:.3f}, sMAPE={m['sMAPE']:.2f}%")
        except Exception as e:
            print(f"  [{i}/{len(tickers)}] {t}: ERROR -> {e}")

    if not all_preds:
        print("No predictions produced.")
        return

    preds_df, metrics_df = pd.concat(all_preds), pd.DataFrame(all_metrics)

    # Write outputs
    paths = [
        (f"{LATEST_PREFIX}/{index_name}/preds.parquet", preds_df),
        (f"{LATEST_PREFIX}/{index_name}/metrics.csv", metrics_df),
        (f"{RUNS_PREFIX}/{index_name}/preds.parquet", preds_df),
        (f"{RUNS_PREFIX}/{index_name}/metrics.csv", metrics_df),
    ]
    for uri, df in paths:
        write_gcs(df, uri)
        print(f"  → wrote {uri}")

def main():
    t0 = time.time()
    print(f"ARIMA job starting… BUCKET={BUCKET}, INDEXES={INDEXES}, H={H}, CAP={CAP}")
    errs = {}
    for idx in INDEXES:
        try:
            run_index(idx)
        except Exception as e:
            errs[idx] = str(e)
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nARIMA job finished in {elapsed/60:.1f} min.")
    if errs and len(errs) == len(INDEXES):
        sys.exit(2)

if __name__ == "__main__":
    main()

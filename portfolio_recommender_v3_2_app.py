
# app.py — Cloud Run (reads from GCS)
# app.py — Cloud Run (reads from GCS)
import os
import time
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Personalized Portfolio", page_icon="📈", layout="wide")
st.title("📈 Personalized Portfolio Builder")

# ============== GCS CONFIG ==============
BUCKET = os.environ.get("GCS_DATA_BUCKET", "fintech-inv-recomm-portfolio-data")
BASE   = os.environ.get("GCS_BASE_PREFIX", "portfolio_data")

def gcs_uri(rel_path: str) -> str:
    return f"gs://{BUCKET}/{BASE}/{rel_path.lstrip('/')}"

# Equity features & prices
SPX_FEATURES      = gcs_uri("sp500/sp500_features_full.parquet")
SPX_PRICES        = gcs_uri("sp500/sp500_prices.parquet")
FTSE100_FEATURES  = gcs_uri("ftse100/ftse100_features_full.parquet")
FTSE100_PRICES    = gcs_uri("ftse100/ftse100_prices.parquet")
FTSE250_FEATURES  = gcs_uri("ftse250/ftse250_features_full.parquet")
FTSE250_PRICES    = gcs_uri("ftse250/ftse250_prices.parquet")

# Optional master files (for names)
SPX_MASTER        = gcs_uri("sp500/sp500_master.csv")
FTSE100_MASTER    = gcs_uri("ftse100/ftse100_master.csv")
FTSE250_MASTER    = gcs_uri("ftse250/ftse250_master.csv")

# ETFs
ETF_FEATURES      = gcs_uri("etf/etf_features.parquet")
ETF_PRICES        = gcs_uri("etf/etf_prices.parquet")
ETF_MASTER        = gcs_uri("etf/etf_master.csv")  # optional; silently ignored if missing

# Sentiment
SENT_CACHE        = gcs_uri("meta/sentiment_sample.parquet")

# ============== UI helpers ==============
def zscore(s):
    s = pd.Series(s).astype(float)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def rotating_status(messages, delay=0.8):
    i = 0
    while True:
        yield messages[i % len(messages)]
        i += 1
        time.sleep(delay)

def alpha_from_horizon(hz: str) -> float:
    if hz == "< 3 years":   return 0.75
    elif hz == "3–5 years": return 0.85
    else:                   return 0.90  # ≥ 5 years

def friendly_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "ticker": "Ticker",
        "name": "Name",
        "asset_type": "Asset",
        "weight": "Weight",
        "alloc_$": "Alloc $",
        "Beta_vs_Benchmark": "Beta",
        "Mom_6M": "6M Return",
        "Mom_12M": "12M Return",
        "Dividend_Yield_TTM": "Dividend Yield",
        "sentiment_z": "Sentiment",
        "final_score": "Score",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

def equity_target_share_from_age(age: int) -> float:
    return float(max(0.2, min(0.9, (100 - age) / 100.0)))

# ============== Robust GCS readers ==============
def _read_parquet(uri: str, label: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(uri)
    except Exception as e:
        st.error(
            f"Missing or unreadable data file for **{label}**:\n\n`{uri}`\n\n"
            f"Error: {e}\n\n"
            "Confirm the file exists in your bucket and that the Cloud Run service "
            "has `GCS_DATA_BUCKET` and `GCS_BASE_PREFIX` set correctly."
        )
        st.stop()

def _read_csv_optional(uri: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(uri)
    except Exception:
        return None  # silently ignore if not present

# ============== Names (master join) ==============
def _extract_name_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a 'name' column if master contains any common name field.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "name"])
    cands = [c for c in ["Name", "Company", "Security", "LongName", "Instrument", "Description"] if c in df.columns]
    name_col = cands[0] if cands else None
    tick_col = "Ticker" if "Ticker" in df.columns else ("ticker" if "ticker" in df.columns else None)
    if not tick_col:
        return pd.DataFrame(columns=["ticker", "name"])
    out = df[[tick_col]].copy()
    out["ticker"] = out[tick_col].astype(str).str.upper()
    out["name"] = df[name_col] if name_col else np.nan
    return out[["ticker", "name"]].drop_duplicates("ticker")

def _attach_names(features_df: pd.DataFrame, which: str) -> pd.DataFrame:
    """
    Merge 'name' onto features using the appropriate master CSV if available.
    """
    masters = {
        "sp500": SPX_MASTER,
        "ftse100": FTSE100_MASTER,
        "ftse250": FTSE250_MASTER,
        "etf": ETF_MASTER
    }
    m_df = _read_csv_optional(masters.get(which, ""))
    names = _extract_name_column(m_df)
    feat = features_df.copy()
    # ensure ticker uppercase before merge
    if "ticker" in feat.columns:
        feat["ticker"] = feat["ticker"].astype(str).str.upper()
    if not names.empty:
        feat = feat.merge(names, on="ticker", how="left")
    else:
        # if features already carry a name-like column, standardize it
        alt = [c for c in ["Name", "Company", "Security", "LongName"] if c in feat.columns]
        if alt:
            feat = feat.rename(columns={alt[0]: "name"})
        else:
            feat["name"] = feat.get("name", np.nan)
    return feat

# ============== Loaders (standardize beta columns) ==============
@st.cache_data(show_spinner=False)
def _load_equity_features(which: str) -> pd.DataFrame:
    if which == "sp500":
        df = _read_parquet(SPX_FEATURES, "SP500 features")
    elif which == "ftse100":
        df = _read_parquet(FTSE100_FEATURES, "FTSE100 features")
    elif which == "ftse250":
        df = _read_parquet(FTSE250_FEATURES, "FTSE250 features")
    else:
        raise ValueError(f"unknown equity bucket: {which}")

    df = df.copy()
    if "Ticker" in df.columns:
        df.rename(columns={"Ticker": "ticker"}, inplace=True)
    df["ticker"] = df["ticker"].astype(str).str.upper()

    # ensure expected cols exist
    for c in ["Mom_6M", "Mom_12M", "Vol_252d", "Dividend_Yield_TTM", "AvgVol_60d"]:
        if c not in df.columns:
            df[c] = np.nan

    # unify beta
    if "Beta_vs_Benchmark" not in df.columns:
        if "Beta_vs_SPY" in df.columns:
            df.rename(columns={"Beta_vs_SPY": "Beta_vs_Benchmark"}, inplace=True)
        elif "Beta_vs_ISF.L" in df.columns:
            df.rename(columns={"Beta_vs_ISF.L": "Beta_vs_Benchmark"}, inplace=True)
        elif "Beta" in df.columns:
            df.rename(columns={"Beta": "Beta_vs_Benchmark"}, inplace=True)
        else:
            df["Beta_vs_Benchmark"] = np.nan

    df["asset_type"] = "Equity"
    # attach names from master
    df = _attach_names(df, which)
    return df

@st.cache_data(show_spinner=False)
def _load_equity_prices(which: str) -> pd.DataFrame:
    if which == "sp500":
        px = _read_parquet(SPX_PRICES, "SP500 prices")
    elif which == "ftse100":
        px = _read_parquet(FTSE100_PRICES, "FTSE100 prices")
    elif which == "ftse250":
        px = _read_parquet(FTSE250_PRICES, "FTSE250 prices")
    else:
        raise ValueError(f"unknown equity price bucket: {which}")

    px = px.copy()
    px["Date"] = pd.to_datetime(px["Date"])
    return px[["Date", "Ticker", "Close"]]

@st.cache_data(show_spinner=False)
def load_universe(index_choice: str):
    if index_choice == "all":
        eq = pd.concat([
            _load_equity_features("sp500"),
            _load_equity_features("ftse100"),
            _load_equity_features("ftse250"),
        ], ignore_index=True).drop_duplicates("ticker")
    else:
        eq = _load_equity_features(index_choice)

    etf = _read_parquet(ETF_FEATURES, "ETF features").copy()
    if "Ticker" in etf.columns:
        etf.rename(columns={"Ticker": "ticker"}, inplace=True)
    etf["ticker"] = etf["ticker"].astype(str).str.upper()

    # attach names for ETFs if master exists
    etf = _attach_names(etf, "etf")

    # ensure expected cols exist
    for c in ["Mom_6M", "Mom_12M", "Vol_252d", "Dividend_Yield_TTM", "AvgVol_60d"]:
        if c not in etf.columns:
            etf[c] = np.nan

    # unify beta
    if "Beta_vs_Benchmark" not in etf.columns:
        if "Beta_vs_SPY" in etf.columns:
            etf.rename(columns={"Beta_vs_SPY": "Beta_vs_Benchmark"}, inplace=True)
        elif "Beta" in etf.columns:
            etf.rename(columns={"Beta": "Beta_vs_Benchmark"}, inplace=True)
        else:
            etf["Beta_vs_Benchmark"] = np.nan

    etf["asset_type"] = "ETF"
    return eq, etf

@st.cache_data(show_spinner=False)
def load_prices(index_choice: str, include_etf: bool = True) -> pd.DataFrame:
    if index_choice == "all":
        px_eq = pd.concat([
            _load_equity_prices("sp500"),
            _load_equity_prices("ftse100"),
            _load_equity_prices("ftse250"),
        ], ignore_index=True)
    else:
        px_eq = _load_equity_prices(index_choice)

    if include_etf:
        px_etf = _read_parquet(ETF_PRICES, "ETF prices").copy()
        px_etf["Date"] = pd.to_datetime(px_etf["Date"])
        px_etf = px_etf[["Date", "Ticker", "Close"]]
        return pd.concat([px_eq, px_etf], ignore_index=True)
    return px_eq

@st.cache_data(show_spinner=False)
def load_sentiment():
    # Accepts many schemas; normalizes to ['ticker','sentiment_z']
    s = _read_parquet(SENT_CACHE, "Sentiment cache (FinBERT)").copy()
    cols = {c.lower(): c for c in s.columns}

    # find ticker column
    tick_col = cols.get("ticker", "Ticker" if "Ticker" in s.columns else s.columns[0])
    # find sentiment score column
    if "sentiment" in cols:
        val_col = cols["sentiment"]
    else:
        cand = [c for c in s.columns if "sentiment" in c.lower() or "score" in c.lower()]
        val_col = cand[0] if cand else (s.columns[1] if len(s.columns) > 1 else s.columns[0])

    s = s.rename(columns={tick_col: "ticker", val_col: "sentiment"})
    s["ticker"] = s["ticker"].astype(str).str.upper()
    s = s.groupby("ticker", as_index=False)["sentiment"].mean()
    s["sentiment_z"] = zscore(s["sentiment"])
    return s[["ticker", "sentiment_z"]]

# ============== Scoring & filtering ==============
def beta_band(risk):
    if risk == "Conservative": return (0.0, 0.8)
    if risk == "Moderate":     return (0.8, 1.4)
    return (1.4, 3.0)

def build_feature_score(df, goal):
    mom = zscore(df["Mom_6M"]).fillna(0)*0.6 + zscore(df["Mom_12M"]).fillna(0)*0.4
    div = zscore(df["Dividend_Yield_TTM"]).fillna(0)
    vol = -zscore(df["Vol_252d"]).fillna(0)
    if goal == "Capital Growth":
        return 0.60*mom + 0.30*vol + 0.10*div
    elif goal == "Dividend Income":
        return 0.50*div + 0.35*vol + 0.15*mom
    return 0.45*mom + 0.35*vol + 0.20*div

def filter_candidates(eq, etf, risk, goal, top_equities=40, top_etfs=10):
    lo, hi = beta_band(risk)
    eqc = eq.copy()
    eqc = eqc[
        (eqc["Beta_vs_Benchmark"].between(lo, hi, inclusive="both")) &
        (eqc["AvgVol_60d"] > 200_000)
    ]
    eqc["feature_score"] = build_feature_score(eqc, goal)
    eqc = eqc.sort_values("feature_score", ascending=False).head(top_equities)

    etf_sel = etf.copy()
    etf_sel = etf_sel[(etf_sel["AvgVol_60d"].fillna(0) > 50_000)]
    etf_sel["feature_score"] = build_feature_score(etf_sel, goal)
    etf_sel = etf_sel.sort_values("feature_score", ascending=False).head(top_etfs)
    return eqc, etf_sel

def blend_with_sentiment(df, sent, alpha: float):
    out = df.merge(sent, how="left", on="ticker")
    out["sentiment_z"] = out["sentiment_z"].fillna(0.0)
    out["feature_z"] = zscore(out["feature_score"])
    out["final_score"] = alpha*out["feature_z"] + (1-alpha)*out["sentiment_z"]
    return out.sort_values("final_score", ascending=False)

# ============== Simple MPT ==============
def max_sharpe_longonly(returns_df, rf=0.015, max_w=0.10, n_iter=6000, seed=42):
    rng = np.random.default_rng(seed)
    X = returns_df.fillna(0).to_numpy()
    mu = X.mean(axis=0) * 252.0
    cov = np.cov(X, rowvar=False) * 252.0

    best = (None, -1e9)
    n = X.shape[1]
    for _ in range(n_iter):
        w = rng.random(n); w = w / w.sum()
        if (w > max_w).any():
            over = (w - max_w).clip(min=0)
            w -= over; w = w / w.sum()
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
        sh = (ret - rf) / (vol + 1e-9)
        if sh > best[1]:
            best = (w, sh)
    return best[0], mu, cov

def rescale_group_weights(weights: pd.Series, group: pd.Series, target_share: float) -> pd.Series:
    w = weights.copy()
    is_equity = (group == "Equity"); is_etf = (group == "ETF")
    w_e = w[is_equity].sum(); w_b = w[is_etf].sum()
    if w_e > 1e-9:
        w.loc[is_equity] *= (target_share / w_e)
    if w_b > 1e-9:
        w.loc[is_etf]    *= ((1.0 - target_share) / w_b)
    return w / w.sum()

# ============== UI ==============
with st.sidebar:
    st.header("Your Preferences")
    index_choice = st.selectbox("Universe", ["SP500", "FTSE100", "FTSE250", "All"], index=0)
    age = st.slider("Age", 18, 80, 32)
    horizon = st.selectbox("Investment Horizon", ["< 3 years", "3–5 years", "≥ 5 years"], index=1)
    risk = st.selectbox("Risk Appetite", ["Conservative", "Moderate", "Aggressive"], index=1)
    goal = st.selectbox("Primary Goal", ["Capital Growth", "Dividend Income", "Balanced"], index=0)
    include_etf = st.toggle("Include ETFs (recommended)", value=True)
    amount = st.number_input("Investment Amount (USD)", min_value=1000.0, value=10000.0, step=500.0)
    st.caption("ETFs help neutralize risk; we include a default top-10 sleeve.")

    # NEW: one-click cache refresh to force fresh reads from GCS
    if st.button("↻ Refresh data from GCS"):
        st.cache_data.clear()
        st.success("Cache cleared. Reloading freshest data from GCS…")
        st.experimental_rerun()

if st.button("Build my portfolio"):
    spinner = st.empty()
    msgs = [
        "We’re creating your personalized portfolio…",
        "BERT is scoring recent news 🙂",
        "Hang on! Almost there…",
        "Crunching features and risk filters…",
    ]
    rot = rotating_status(msgs, delay=0.9); spinner.info(next(rot))

    idx_map = {"SP500": "sp500", "FTSE100": "ftse100", "FTSE250": "ftse250", "All": "all"}
    idx = idx_map[index_choice]

    eq, etf = load_universe(idx); spinner.info(next(rot))
    sent = load_sentiment();      spinner.info(next(rot))
    px = load_prices(idx, include_etf=include_etf)

    # candidates
    eq_40, etf_10 = filter_candidates(eq, etf, risk, goal, top_equities=40, top_etfs=10 if include_etf else 0)

    alpha = alpha_from_horizon(horizon)
    cand = pd.concat([eq_40, etf_10], ignore_index=True)
    blended = blend_with_sentiment(cand, sent, alpha=alpha)

    spinner.info(next(rot))
    use_tickers = blended["ticker"].tolist()
    mat = (px.query("Ticker in @use_tickers")[["Date","Ticker","Close"]]
           .pivot(index="Date", columns="Ticker", values="Close")
           .sort_index()
           .pct_change(fill_method=None)
           .dropna(how="all"))
    mat = mat.dropna(axis=1, thresh=int(0.6*len(mat)))
    keep = [t for t in blended["ticker"] if t in mat.columns]
    blended = blended[blended["ticker"].isin(keep)].reset_index(drop=True)
    mat = mat[keep]

    spinner.info("Optimizing weights (MPT)…")
    if mat.shape[1] < 5:
        st.warning("Too few instruments after filtering/price alignment."); st.stop()

    w0, mu, cov = max_sharpe_longonly(mat, rf=0.015, max_w=0.10, n_iter=6000, seed=42)

    target_equity_share = equity_target_share_from_age(age)
    w_series = pd.Series(w0, index=mat.columns)
    types = blended.set_index("ticker")["asset_type"].reindex(mat.columns).fillna("Equity")
    w_adj = rescale_group_weights(w_series, types, target_share=target_equity_share)

    res = blended.set_index("ticker").loc[mat.columns].reset_index()
    res["weight"] = w_adj.values
    res["alloc_$"] = (res["weight"] * amount).round(2)

    # Portfolio beta (weighted Beta_vs_Benchmark)
    res["Beta_contrib"] = res["weight"] * res["Beta_vs_Benchmark"].fillna(1.0)
    port_beta = float(res["Beta_contrib"].sum())

    spinner.success("Portfolio ready!")

    eq_n = int((res["asset_type"] == "Equity").sum())
    etf_n = int((res["asset_type"] == "ETF").sum())
    st.subheader(
        f"Summary: {eq_n} Equities, {etf_n} ETFs · Target Equity Share ≈ {target_equity_share:.0%} · Portfolio Beta ≈ {port_beta:.2f}"
    )

    # Show Name next to Ticker
    show_cols = [
        "ticker", "name", "asset_type", "weight", "alloc_$",
        "Beta_vs_Benchmark", "Mom_6M", "Mom_12M",
        "Dividend_Yield_TTM", "sentiment_z", "final_score"
    ]
    present = [c for c in show_cols if c in res.columns]
    pretty = friendly_cols(res[present])

    st.dataframe(
        pretty.style.format({
            "Weight": "{:.2%}", "Alloc $": "${:,.0f}",
            "Beta": "{:.2f}",
            "6M Return": "{:.2%}", "12M Return": "{:.2%}",
            "Dividend Yield": "{:.2%}", "Sentiment": "{:.2f}",
            "Score": "{:.2f}",
        }),
        use_container_width=True
    )

    # Data-health quick check (collapsed)
    def summarize_beta(df: pd.DataFrame, label: str):
        total = len(df)
        miss = int(df["Beta_vs_Benchmark"].isna().sum()) if "Beta_vs_Benchmark" in df.columns else total
        st.write(f"**{label}** → missing Beta: {miss} / {total} ({(miss/total if total else 0):.1%})")

    with st.expander("Data health (debug)", expanded=False):
        if idx == "all":
            e_spx  = _load_equity_features("sp500")
            e_f100 = _load_equity_features("ftse100")
            e_f250 = _load_equity_features("ftse250")
            summarize_beta(e_spx,  "SP500")
            summarize_beta(e_f100, "FTSE100")
            summarize_beta(e_f250, "FTSE250")
        else:
            summarize_beta(eq, idx.upper())

    st.caption(
        "Notes: long-only; per-name cap 10%; equity/ETF split uses (100 − age) rule; "
        "β is Beta_vs_Benchmark from your feature files. Data is loaded from GCS."
    )

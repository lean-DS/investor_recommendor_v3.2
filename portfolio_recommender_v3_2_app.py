# #portfolio_recommender_v3_2_app.py — Cloud Run (reads from GCS)
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import auth

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Personalized Portfolio", page_icon="📈", layout="wide")
st.title("Personalized Portfolio Builder v3.2.0")

auth.handle_oauth_callback()

# ===== DB init =====
import db  # your db.py

@st.cache_resource
def _init_db():
    db.ensure_schema()
    # Call enable_rls() only if you actually implemented it.
    if hasattr(db, "enable_rls"):
        try:
            db.enable_rls()
        except Exception:
            pass
    return True

DB_READY = _init_db()

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

# ETFs
ETF_FEATURES      = gcs_uri("etf/etf_features.parquet")
ETF_PRICES        = gcs_uri("etf/etf_prices.parquet")

# Sentiment
SENT_CACHE        = gcs_uri("meta/sentiment_sample.parquet")

# ============== Helpers ==============
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
    if hz == "3–5 years":   return 0.85
    return 0.90  # ≥ 5 years

def equity_target_share_from_age(age: int) -> float:
    return float(max(0.2, min(0.9, (100 - age) / 100.0)))

def risk_recommendation_from_age(age: int) -> str:
    if age < 30:        return "Aggressive"
    if 30 <= age <= 45: return "Moderate"
    return "Conservative"

def friendly_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Show 'Ticker' to the user as 'Securities'
    rename_map = {
        "Ticker": "Securities",
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

# ============== Readers ==============
def _read_parquet(uri: str, label: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(uri)
    except Exception as e:
        st.error(
            f"Missing or unreadable data file for **{label}**:\n\n`{uri}`\n\nError: {e}\n\n"
            "Confirm the file exists in your bucket and that Cloud Run has "
            "`GCS_DATA_BUCKET` and `GCS_BASE_PREFIX` set correctly."
        )
        st.stop()

@st.cache_data(show_spinner=False)
def _load_equity_features(which: str) -> pd.DataFrame:
    # Keep 'Ticker' as-is; do NOT rename to 'ticker'
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
        df["Ticker"] = df["Ticker"].astype(str).str.upper()

    # ensure expected cols exist
    for c in ["Mom_6M", "Mom_12M", "Vol_252d", "Dividend_Yield_TTM", "AvgVol_60d"]:
        if c not in df.columns:
            df[c] = np.nan

    # unify beta column name (to Beta_vs_Benchmark)
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
        ], ignore_index=True).drop_duplicates("Ticker")
    else:
        eq = _load_equity_features(index_choice)

    etf = _read_parquet(ETF_FEATURES, "ETF features").copy()
    if "Ticker" in etf.columns:
        etf["Ticker"] = etf["Ticker"].astype(str).str.upper()

    # ensure expected cols exist
    for c in ["Mom_6M", "Mom_12M", "Vol_252d", "Dividend_Yield_TTM", "AvgVol_60d"]:
        if c not in etf.columns:
            etf[c] = np.nan

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
    # Normalize to 'Ticker' (uppercase)
    s = _read_parquet(SENT_CACHE, "Sentiment cache (FinBERT)").copy()
    cols = {c.lower(): c for c in s.columns}

    tick_col = cols.get("ticker", "Ticker" if "Ticker" in s.columns else s.columns[0])
    if "sentiment" in cols:
        val_col = cols["sentiment"]
    else:
        cand = [c for c in s.columns if "sentiment" in c.lower() or "score" in c.lower()]
        val_col = cand[0] if cand else (s.columns[1] if len(s.columns) > 1 else s.columns[0])

    s = s.rename(columns={tick_col: "Ticker", val_col: "sentiment"})
    s["Ticker"] = s["Ticker"].astype(str).str.upper()
    s = s.groupby("Ticker", as_index=False)["sentiment"].mean()
    s["sentiment_z"] = zscore(s["sentiment"])
    return s[["Ticker", "sentiment_z"]]

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
    out = df.merge(sent, how="left", on="Ticker")
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
    goal = st.selectbox("Primary Goal", ["Capital Growth", "Dividend Income", "Balanced"], index=0)
    amount = st.number_input("Investment Amount (USD)", min_value=1000.0, value=10000.0, step=500.0)

    # Recommend risk but let user choose (default = recommendation)
    recommended = risk_recommendation_from_age(age)
    risk = st.selectbox(
        "Risk Appetite",
        ["Conservative", "Moderate", "Aggressive"],
        index=["Conservative", "Moderate", "Aggressive"].index(recommended),
        help="Default set from age; feel free to adjust.",
    )
    st.caption(f"Recommended from age: **{recommended}**")

    st.divider()
    st.caption("Account")
    # Render login / profile; this stores the canonical UUID in session on success
    auth.login_section()

# Canonical UUID for RLS-aware DB calls (None if not signed-in)
uid = auth.get_current_uid()

if uid:
    st.caption("Successfully Signed In")
else:
    st.caption("You’re not signed in, you can build, but must sign in to save.")
        
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
    px = load_prices(idx, include_etf=True)  # ETFs always included

    # candidates
    eq_40, etf_10 = filter_candidates(eq, etf, risk, goal, top_equities=40, top_etfs=10)

    alpha = alpha_from_horizon(horizon)
    cand = pd.concat([eq_40, etf_10], ignore_index=True)
    blended = blend_with_sentiment(cand, sent, alpha=alpha)

    spinner.info(next(rot))
    use_tickers = blended["Ticker"].tolist()
    mat = (px.query("Ticker in @use_tickers")[["Date","Ticker","Close"]]
           .pivot(index="Date", columns="Ticker", values="Close")
           .sort_index()
           .pct_change(fill_method=None)
           .dropna(how="all"))
    mat = mat.dropna(axis=1, thresh=int(0.6*len(mat)))
    keep = [t for t in blended["Ticker"] if t in mat.columns]
    blended = blended[blended["Ticker"].isin(keep)].reset_index(drop=True)
    mat = mat[keep]

    spinner.info("Optimizing weights (MPT)…")
    if mat.shape[1] < 5:
        st.warning("Too few instruments after filtering/price alignment."); st.stop()

    w0, mu, cov = max_sharpe_longonly(mat, rf=0.015, max_w=0.10, n_iter=6000, seed=42)

    target_equity_share = equity_target_share_from_age(age)
    w_series = pd.Series(w0, index=mat.columns)
    types = blended.set_index("Ticker")["asset_type"].reindex(mat.columns).fillna("Equity")
    w_adj = rescale_group_weights(w_series, types, target_share=target_equity_share)

    res = blended.set_index("Ticker").loc[mat.columns].reset_index()
    res["weight"] = w_adj.values
    res["alloc_$"] = (res["weight"] * amount).round(2)

    # Portfolio beta (weighted Beta_vs_Benchmark)
    res["Beta_contrib"] = res["weight"] * res["Beta_vs_Benchmark"].fillna(1.0)
    port_beta = float(res["Beta_contrib"].sum())

    spinner.success("Portfolio ready!")

    eq_n = int((res["asset_type"] == "Equity").sum())
    etf_n = int((res["asset_type"] == "ETF").sum())
    st.subheader(
        f"Summary: {eq_n} Equities, {etf_n} ETFs · "
        f"Target Equity Share ≈ {target_equity_share:.0%} · Portfolio Beta ≈ {port_beta:.2f}"
    )

    # Display (use 'Securities' label for the Ticker column)
    show_cols = [
        "Ticker", "asset_type", "weight", "alloc_$",
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

    # Persist the latest build to session for the Save section
    st.session_state["last_res"] = res
    st.session_state["last_meta"] = {
        "index_choice": idx.upper(),
        "risk_profile": risk,        # user-chosen risk
        "horizon": horizon,
        "amount_usd": float(amount),
    }


# ---- Save portfolio to DB ----
st.subheader("Save portfolio")
if "last_res" in st.session_state and "last_meta" in st.session_state:
    default_name = f'{st.session_state["last_meta"]["index_choice"]} • {st.session_state["last_meta"]["risk_profile"]} • {pd.Timestamp.utcnow().date().isoformat()}'
    pf_name = st.text_input("Portfolio name", value=default_name)

    if st.button("💾 Save to database", use_container_width=True, type="primary", disabled=not bool(uid)):
        if not uid:
            st.warning("Please sign in to save your portfolio.")
        else:
            try:
                meta = dict(st.session_state["last_meta"])
                meta["name"] = pf_name
                portfolio_id = db.save_portfolio(uid, meta, st.session_state["last_res"])
                st.success(f"Saved! Portfolio ID: {portfolio_id}")
            except Exception as e:
                st.error(f"Save failed: {e}")
    elif not uid:
        st.info("Sign in to save.")
else:
    st.info("Build a portfolio first to enable saving.")
    
# ---- List my saved portfolios ----
st.subheader("My portfolios")
try:
    if uid:
        pf = db.list_user_portfolios(uid)
        if pf.empty:
            st.info("No saved portfolios yet.")
        else:
            st.dataframe(pf, use_container_width=True)
    else:
        st.info("Sign in to view your saved portfolios.")
except Exception as e:
    st.error(f"List failed: {e}")

st.caption(
    "Notes: long-only; per-name cap 10%; equity/ETF split uses (100 − age) rule; "
    "β is Beta_vs_Benchmark from your feature files. Data is loaded from GCS."
)

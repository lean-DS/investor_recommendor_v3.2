# db.py
import os
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Any, Optional

from google.cloud.sql.connector import Connector, IPTypes
import sqlalchemy

# ---------- Cloud SQL connector (singleton) ----------
_CONNECTOR: Optional[Connector] = None

def _get_connector() -> Connector:
    global _CONNECTOR
    if _CONNECTOR is None:
        _CONNECTOR = Connector()
    return _CONNECTOR

def _creator():
    # INSTANCE_CONNECTION_NAME like "project:region:instance"
    connector = _get_connector()
    return connector.connect(
        os.environ["INSTANCE_CONNECTION_NAME"],
        "pg8000",
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASS"],
        db=os.environ["DB_NAME"],
        ip_type=IPTypes.PRIVATE if os.getenv("DB_PRIVATE_IP") == "1" else IPTypes.PUBLIC,
    )

# Optional: call this at process exit if you want explicit cleanup
def close_connector():
    global _CONNECTOR
    if _CONNECTOR is not None:
        _CONNECTOR.close()
        _CONNECTOR = None

# ---------- SQLAlchemy engine (singleton) ----------
_ENGINE = None
def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=_creator,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
    return _ENGINE

# ---------- DDL ----------
_DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS users (
      user_id TEXT PRIMARY KEY,
      email   TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolios (
      portfolio_id   BIGSERIAL PRIMARY KEY,
      user_id        TEXT NOT NULL REFERENCES users(user_id),
      created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
      index_choice   TEXT,
      risk_profile   TEXT,
      horizon        TEXT,
      amount_usd     NUMERIC
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS holdings (
      holding_id     BIGSERIAL PRIMARY KEY,
      portfolio_id   BIGINT NOT NULL REFERENCES portfolios(portfolio_id) ON DELETE CASCADE,
      ticker         TEXT NOT NULL,
      asset_type     TEXT,
      weight         DOUBLE PRECISION,
      alloc_usd      NUMERIC,
      beta           DOUBLE PRECISION,
      mom_6m         DOUBLE PRECISION,
      mom_12m        DOUBLE PRECISION,
      div_yield_ttm  DOUBLE PRECISION,
      sentiment_z    DOUBLE PRECISION,
      final_score    DOUBLE PRECISION
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_holdings_portfolio ON holdings(portfolio_id);",
    "CREATE INDEX IF NOT EXISTS idx_holdings_ticker   ON holdings(ticker);",
]

def ensure_schema():
    eng = get_engine()
    with eng.begin() as con:
        for stmt in _DDL_STATEMENTS:
            con.exec_driver_sql(stmt)

# ---------- Helpers ----------
def _to_decimal(x: Any) -> Optional[Decimal]:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    try:
        return Decimal(str(x))
    except Exception:
        return None

def _to_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return None
        return xf
    except Exception:
        return None

# ---------- CRUD ----------
def upsert_user(user_dict: dict):
    """
    user_dict: {"uid": "<auth uid>", "email": "user@x.com"}
    """
    eng = get_engine()
    with eng.begin() as con:
        con.exec_driver_sql(
            """
            INSERT INTO users(user_id, email)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE SET email = EXCLUDED.email
            """,
            (user_dict["uid"], user_dict.get("email")),
        )

def save_portfolio(user_dict: dict, meta: dict, res_df: pd.DataFrame) -> int:
    """
    meta: {"index_choice": "...", "risk": "...", "horizon": "...", "amount": float}
    res_df: portfolio rows with columns:
      ticker, asset_type, weight, alloc_$, Beta_vs_Benchmark, Mom_6M, Mom_12M,
      Dividend_Yield_TTM, sentiment_z, final_score
    """
    eng = get_engine()
    with eng.begin() as con:
        pid = con.exec_driver_sql(
            """
            INSERT INTO portfolios(user_id, index_choice, risk_profile, horizon, amount_usd)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING portfolio_id
            """,
            (
                user_dict["uid"],
                meta.get("index_choice"),
                meta.get("risk"),
                meta.get("horizon"),
                _to_decimal(meta.get("amount")),
            ),
        ).scalar_one()

        # Insert holdings (NULLs instead of zeros where missing)
        cols = {
            "ticker": "ticker",
            "asset_type": "asset_type",
            "weight": "weight",
            "alloc_$": "alloc_usd",
            "Beta_vs_Benchmark": "beta",
            "Mom_6M": "mom_6m",
            "Mom_12M": "mom_12m",
            "Dividend_Yield_TTM": "div_yield_ttm",
            "sentiment_z": "sentiment_z",
            "final_score": "final_score",
        }

        for _, r in res_df.iterrows():
            con.exec_driver_sql(
                """
                INSERT INTO holdings(
                  portfolio_id, ticker, asset_type, weight, alloc_usd, beta,
                  mom_6m, mom_12m, div_yield_ttm, sentiment_z, final_score
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    pid,
                    r.get("ticker"),
                    r.get("asset_type"),
                    _to_float_or_none(r.get("weight")),
                    _to_decimal(r.get("alloc_$")),
                    _to_float_or_none(r.get("Beta_vs_Benchmark")),
                    _to_float_or_none(r.get("Mom_6M")),
                    _to_float_or_none(r.get("Mom_12M")),
                    _to_float_or_none(r.get("Dividend_Yield_TTM")),
                    _to_float_or_none(r.get("sentiment_z")),
                    _to_float_or_none(r.get("final_score")),
                ),
            )
    return pid

def list_user_portfolios(uid: str) -> pd.DataFrame:
    q = """
      SELECT p.portfolio_id,
             p.created_at,
             p.index_choice,
             p.risk_profile,
             p.horizon,
             p.amount_usd,
             COUNT(h.holding_id) AS n_holdings
      FROM portfolios p
      LEFT JOIN holdings h ON h.portfolio_id = p.portfolio_id
      WHERE p.user_id = %s
      GROUP BY p.portfolio_id
      ORDER BY p.created_at DESC
    """
    return pd.read_sql(q, get_engine(), params=(uid,))

# db.py
import os
import numpy as np
import pandas as pd
from google.cloud.sql.connector import Connector, IPTypes
import sqlalchemy

def _creator():
    connector = Connector()
    return connector.connect(
        os.environ["INSTANCE_CONNECTION_NAME"],  # project:region:instance
        "pg8000",  # or "psycopg2"
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASS"],
        db=os.environ["DB_NAME"],
        ip_type=IPTypes.PRIVATE if os.getenv("DB_PRIVATE_IP")=="1" else IPTypes.PUBLIC,
    )

_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=_creator,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
    return _engine

def ensure_schema():
    eng = get_engine()
    with eng.begin() as con:
        con.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS users (
          user_id TEXT PRIMARY KEY,
          email   TEXT
        );
        CREATE TABLE IF NOT EXISTS portfolios (
          portfolio_id   BIGSERIAL PRIMARY KEY,
          user_id        TEXT NOT NULL REFERENCES users(user_id),
          created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
          index_choice   TEXT,
          risk_profile   TEXT,
          horizon        TEXT,
          amount_usd     NUMERIC
        );
        CREATE TABLE IF NOT EXISTS holdings (
          holding_id    BIGSERIAL PRIMARY KEY,
          portfolio_id  BIGINT NOT NULL REFERENCES portfolios(portfolio_id) ON DELETE CASCADE,
          ticker        TEXT NOT NULL,
          asset_type    TEXT,
          weight        DOUBLE PRECISION,
          alloc_usd     NUMERIC,
          beta          DOUBLE PRECISION,
          mom_6m        DOUBLE PRECISION,
          mom_12m       DOUBLE PRECISION,
          div_yield_ttm DOUBLE PRECISION,
          sentiment_z   DOUBLE PRECISION,
          final_score   DOUBLE PRECISION
        );
        CREATE INDEX IF NOT EXISTS idx_holdings_portfolio ON holdings(portfolio_id);
        """)

def upsert_user(user_dict: dict):
    eng = get_engine()
    with eng.begin() as con:
        con.exec_driver_sql(
            "INSERT INTO users(user_id,email) VALUES (%s,%s) "
            "ON CONFLICT (user_id) DO UPDATE SET email=EXCLUDED.email",
            (user_dict["uid"], user_dict.get("email"))
        )

def save_portfolio(user_dict: dict, meta: dict, res_df: pd.DataFrame) -> int:
    eng = get_engine()
    with eng.begin() as con:
        pid = con.exec_driver_sql(
            "INSERT INTO portfolios(user_id,index_choice,risk_profile,horizon,amount_usd) "
            "VALUES (%s,%s,%s,%s,%s) RETURNING portfolio_id",
            (user_dict["uid"], meta["index_choice"], meta["risk"], meta["horizon"], meta["amount"])
        ).scalar_one()

        rows = res_df.fillna(np.nan).to_dict("records")
        for r in rows:
            con.exec_driver_sql("""
                INSERT INTO holdings(
                  portfolio_id,ticker,asset_type,weight,alloc_usd,beta,
                  mom_6m,mom_12m,div_yield_ttm,sentiment_z,final_score
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                pid, r.get("ticker"), r.get("asset_type"),
                float(r.get("weight",0)), float(r.get("alloc_$",0)),
                float(r.get("Beta_vs_Benchmark") or 0),
                float(r.get("Mom_6M") or 0), float(r.get("Mom_12M") or 0),
                float(r.get("Dividend_Yield_TTM") or 0),
                float(r.get("sentiment_z") or 0), float(r.get("final_score") or 0),
            ))
    return pid

def list_user_portfolios(uid: str) -> pd.DataFrame:
    q = """
      SELECT p.portfolio_id, p.created_at, p.index_choice, p.risk_profile, p.horizon, p.amount_usd,
             count(h.holding_id) AS n_holdings
      FROM portfolios p
      LEFT JOIN holdings h ON h.portfolio_id = p.portfolio_id
      WHERE p.user_id = %s
      GROUP BY p.portfolio_id
      ORDER BY p.created_at DESC
    """
    return pd.read_sql(q, get_engine(), params=(uid,))

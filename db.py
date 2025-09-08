# db.py
# deps (requirements.txt):
#   google-cloud-sql-connector[pg8000]>=1.10
#   SQLAlchemy>=2.0
#   pandas>=2.1
#   pg8000>=1.30

import os
import numpy as np
import pandas as pd
from typing import Optional
from google.cloud.sql.connector import Connector, IPTypes
import sqlalchemy

import db  # your db.py

@st.cache_resource
def _init_db():
    db.ensure_schema()
    db.enable_rls()
    return True

DB_READY = _init_db()

# ---------- Connection ----------

_connector: Optional[Connector] = None
_engine: Optional[sqlalchemy.Engine] = None

def _creator():
    global _connector
    if _connector is None:
        _connector = Connector()

    return _connector.connect(
        os.environ["INSTANCE_CONNECTION_NAME"],   # "project:region:instance"
        "pg8000",
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASS"],
        db=os.environ["DB_NAME"],
        ip_type=IPTypes.PRIVATE if os.getenv("DB_PRIVATE_IP") == "1" else IPTypes.PUBLIC,
    )

def get_engine() -> sqlalchemy.Engine:
    global _engine
    if _engine is None:
        _engine = sqlalchemy.create_engine(
            "postgresql+pg8000://", creator=_creator,
            pool_pre_ping=True, pool_recycle=1800
        )
    return _engine

# ---------- One-time setup ----------

def ensure_schema():
    """
    Creates extensions, tables, constraints, and helpful indexes.
    Safe to run multiple times.
    """
    eng = get_engine()
    with eng.begin() as con:
        # uuid generator
        con.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

        # users
        con.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS app_user (
          id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          email TEXT UNIQUE
        );
        """)

        # portfolios
        con.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS portfolio (
          id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          user_id      UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
          name         TEXT NOT NULL,
          created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
          index_choice TEXT,
          risk_profile TEXT,
          horizon      TEXT,
          amount_usd   NUMERIC
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_portfolio_user_name
          ON portfolio(user_id, name);
        CREATE INDEX IF NOT EXISTS idx_portfolio_user
          ON portfolio(user_id);
        """)

        # holdings (positions)
        con.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS position (
          id            BIGSERIAL PRIMARY KEY,
          portfolio_id  UUID NOT NULL REFERENCES portfolio(id) ON DELETE CASCADE,
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
        CREATE INDEX IF NOT EXISTS idx_position_portfolio ON position(portfolio_id);
        CREATE INDEX IF NOT EXISTS idx_position_ticker    ON position(ticker);
        """)

def enable_rls():
    """
    Enables row-level security and policies that isolate data by app user.
    Your app must set:  SET LOCAL app.user_id = '<uuid>'  per request/txn.
    """
    eng = get_engine()
    with eng.begin() as con:
        con.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        # RLS on portfolio/position
        con.exec_driver_sql("ALTER TABLE portfolio ENABLE ROW LEVEL SECURITY;")
        con.exec_driver_sql("ALTER TABLE position  ENABLE ROW LEVEL SECURITY;")

        # Allow owner (matching app.user_id) to see their rows
        con.exec_driver_sql("""
        DROP POLICY IF EXISTS portfolio_isolation ON portfolio;
        CREATE POLICY portfolio_isolation
        ON portfolio
        USING (user_id::text = current_setting('app.user_id', true));
        """)
        con.exec_driver_sql("""
        DROP POLICY IF EXISTS position_isolation ON position;
        CREATE POLICY position_isolation
        ON position
        USING (portfolio_id IN (
          SELECT id FROM portfolio
          WHERE user_id::text = current_setting('app.user_id', true)
        ));
        """)

def _set_rls_user(con: sqlalchemy.Connection, uid: str):
    """Call this at the start of each request/transaction."""
    con.exec_driver_sql("SET LOCAL app.user_id = %s", (uid,))

# ---------- CRUD helpers ----------

def upsert_user(uid: str, email: Optional[str]):
    """
    Insert or update a user record (id=email pair).
    """
    eng = get_engine()
    with eng.begin() as con:
        # ensure a row exists with exact id/email
        con.exec_driver_sql("""
        INSERT INTO app_user(id, email)
        VALUES (%s, %s)
        ON CONFLICT (id) DO UPDATE SET email = EXCLUDED.email
        """, (uid, email))

def save_portfolio(uid: str, meta: dict, res_df: pd.DataFrame, portfolio_name: Optional[str] = None) -> str:
    """
    Persist a portfolio header + its holdings for the given user.
    Returns portfolio_id (UUID).
    meta expects: index_choice, risk_profile, horizon, amount_usd
    res_df expects columns: ['ticker','asset_type','weight','alloc_$','Beta_vs_Benchmark',
                             'Mom_6M','Mom_12M','Dividend_Yield_TTM','sentiment_z','final_score']
    """
    name = portfolio_name or meta.get("name") or "My Portfolio"
    eng = get_engine()
    with eng.begin() as con:
        _set_rls_user(con, uid)

        pid = con.exec_driver_sql("""
          INSERT INTO portfolio(user_id, name, index_choice, risk_profile, horizon, amount_usd)
          VALUES (%s,%s,%s,%s,%s,%s) RETURNING id
        """, (
            uid,
            name,
            meta.get("index_choice"),
            meta.get("risk_profile"),
            meta.get("horizon"),
            meta.get("amount_usd"),
        )).scalar_one()

        # Insert holdings
        for r in res_df.fillna(np.nan).to_dict("records"):
            con.exec_driver_sql("""
              INSERT INTO position(
                portfolio_id, ticker, asset_type, weight, alloc_usd, beta,
                mom_6m, mom_12m, div_yield_ttm, sentiment_z, final_score
              )
              VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                pid,
                r.get("ticker"),
                r.get("asset_type"),
                float(r.get("weight") or 0),
                float(r.get("alloc_$") or 0),
                float(r.get("Beta_vs_Benchmark") or 0),
                float(r.get("Mom_6M") or 0),
                float(r.get("Mom_12M") or 0),
                float(r.get("Dividend_Yield_TTM") or 0),
                float(r.get("sentiment_z") or 0),
                float(r.get("final_score") or 0),
            ))
    return str(pid)

def list_user_portfolios(uid: str) -> pd.DataFrame:
    eng = get_engine()
    with eng.begin() as con:
        _set_rls_user(con, uid)
        q = """
          SELECT p.id AS portfolio_id, p.name, p.created_at,
                 p.index_choice, p.risk_profile, p.horizon, p.amount_usd,
                 count(o.id) AS n_holdings
          FROM portfolio p
          LEFT JOIN position o ON o.portfolio_id = p.id
          GROUP BY p.id
          ORDER BY p.created_at DESC
        """
        return pd.read_sql(q, con)

def get_portfolio_positions(uid: str, portfolio_id: str) -> pd.DataFrame:
    eng = get_engine()
    with eng.begin() as con:
        _set_rls_user(con, uid)
        q = """
          SELECT ticker, asset_type, weight, alloc_usd, beta,
                 mom_6m, mom_12m, div_yield_ttm, sentiment_z, final_score
          FROM position
          WHERE portfolio_id = %s
          ORDER BY weight DESC NULLS LAST
        """
        return pd.read_sql(q, con, params=(portfolio_id,))

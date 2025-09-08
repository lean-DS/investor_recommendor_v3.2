# auth.py
import os, json
import streamlit as st
import firebase_admin
from firebase_admin import auth as fb_auth, credentials

# One-time Firebase Admin init from env var FIREBASE_CREDS (service account JSON)
if not firebase_admin._apps:
    _creds_json = os.environ.get("FIREBASE_CREDS")
    if not _creds_json:
        raise RuntimeError("FIREBASE_CREDS env var not set. Provide your Firebase service account JSON.")
    cred = credentials.Certificate(json.loads(_creds_json))
    firebase_admin.initialize_app(cred)

@st.cache_resource(show_spinner=False)
def verify_id_token(id_token: str):
    """Verify Firebase ID token and return decoded claims or None."""
    try:
        return fb_auth.verify_id_token(id_token, clock_skew_seconds=60)
    except Exception:
        return None

def sidebar_login() -> dict | None:
    """
    Minimal sign-in UX for now: paste ID token (replace with Firebase Web SDK later).
    Returns decoded user dict or None.
    """
    with st.sidebar:
        st.markdown("### Sign in")
        token = st.text_input("Paste Firebase ID token", type="password")
        user = verify_id_token(token) if token else None
        if user:
            st.success(f"Signed in as {user.get('email','user')}")
        else:
            st.info("Not signed in")
        return user

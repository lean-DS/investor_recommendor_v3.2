# auth.py
import os
import time
import secrets
import requests
import streamlit as st
from urllib.parse import urlencode

import db  # uses upsert_user(...) and returns canonical UUID

AUTH_BASE   = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL   = "https://oauth2.googleapis.com/token"
USERINFO    = "https://openidconnect.googleapis.com/v1/userinfo"
SCOPES      = ["openid", "email", "profile"]

# Required env vars (set these in Cloud Run)
CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
REDIRECT_URI  = os.environ.get("OAUTH_REDIRECT_URI")

def _require_env():
    missing = [k for k,v in {
        "GOOGLE_CLIENT_ID": CLIENT_ID,
        "GOOGLE_CLIENT_SECRET": CLIENT_SECRET,
        "OAUTH_REDIRECT_URI": REDIRECT_URI,
    }.items() if not v]
    if missing:
        st.error(f"Missing environment variables: {', '.join(missing)}")
        st.stop()

def login_button(label="Sign in with Google"):
    """Render a link-style button to initiate OAuth."""
    _require_env()
    # generate and remember state for CSRF protection
    state = secrets.token_urlsafe(24)
    st.session_state["oauth_state"] = state

    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent",
        "state": state,
    }
    st.link_button(label, f"{AUTH_BASE}?{urlencode(params)}", use_container_width=True)

def _exchange_code_for_tokens(code: str):
    data = {
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    r = requests.post(TOKEN_URL, data=data, timeout=20)
    r.raise_for_status()
    return r.json()

def _fetch_userinfo(access_token: str):
    r = requests.get(USERINFO, headers={"Authorization": f"Bearer {access_token}"}, timeout=20)
    r.raise_for_status()
    return r.json()

def handle_oauth_callback():
    """
    Call this very early in your app file (before using user info),
    e.g. right after imports, to capture ?code=...&state=...
    """
    _require_env()

    # Streamlit 1.38 keeps experimental_get_query_params; use it for compatibility
    q = st.experimental_get_query_params()
    code  = (q.get("code") or [None])[0]
    state = (q.get("state") or [None])[0]
    if not code:
        return  # nothing to do

    # Validate state
    saved_state = st.session_state.get("oauth_state")
    if not saved_state or state != saved_state:
        st.error("OAuth state mismatch. Please try signing in again.")
        # clean query params regardless
        st.experimental_set_query_params()
        return

    try:
        tokens = _exchange_code_for_tokens(code)
        info   = _fetch_userinfo(tokens["access_token"])

        # Google OIDC fields
        google_sub = info.get("sub")
        email      = info.get("email")
        name       = info.get("name")
        picture    = info.get("picture")

        # Ensure user exists in DB and get canonical UUID
        app_user_id = db.upsert_user(
            uid=None,                 # let DB choose; we key by google_sub
            email=email,
            google_sub=google_sub,
            name=name,
        )

        # Minimal session payload (store canonical UUID for RLS)
        st.session_state["user"] = {
            "app_user_id": app_user_id,           # UUID used for RLS
            "google_sub": google_sub,
            "email": email,
            "name": name,
            "picture": picture,
            "ts": int(time.time()),
            "tokens": {"access_token": tokens.get("access_token")},
        }

        # Clean URL (remove ?code=..., ?state=...)
        st.experimental_set_query_params()
        st.success(f"Signed in as {email}")

    except Exception as e:
        st.error(f"Google sign-in failed: {e}")
        # Clean the URL to avoid repeated attempts with same code
        st.experimental_set_query_params()

def get_current_user():
    """Return the user dict from session or None."""
    return st.session_state.get("user")

def get_current_uid() -> str | None:
    """Return canonical app_user UUID to use with DB & RLS, or None if not signed in."""
    u = get_current_user()
    return u.get("app_user_id") if u else None

def logout_button():
    if st.button("Sign out"):
        st.session_state.pop("user", None)
        st.session_state.pop("oauth_state", None)
        st.experimental_set_query_params()
        st.success("Signed out")

def login_section():
    """
    Small convenience helper to render a login area in your sidebar or header.
    """
    u = get_current_user()
    if u:
        col1, col2 = st.columns([1,4])
        with col1:
            if u.get("picture"):
                st.image(u["picture"], width=48)
        with col2:
            st.markdown(f"**{u.get('name') or u.get('email')}**")
            st.caption(u.get("email"))
        logout_button()
    else:
        login_button("Sign in with Google")

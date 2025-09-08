# auth.py
import os, time, json, requests, streamlit as st
from urllib.parse import urlencode

AUTH_BASE = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO  = "https://openidconnect.googleapis.com/v1/userinfo"
SCOPES = ["openid","email","profile"]

CLIENT_ID     = os.environ["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
REDIRECT_URI  = os.environ["OAUTH_REDIRECT_URI"]

def login_button(label="Sign in with Google"):
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent",
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
    # Call this at the top of app.py to capture ?code=...
    q = st.experimental_get_query_params()
    code = q.get("code", [None])[0]
    if not code:
        return
    try:
        tokens = _exchange_code_for_tokens(code)
        info = _fetch_userinfo(tokens["access_token"])
        # Minimal session payload
        st.session_state["user"] = {
            "uid": info.get("sub"),
            "email": info.get("email"),
            "name": info.get("name"),
            "picture": info.get("picture"),
            "ts": int(time.time()),
            "tokens": {"access_token": tokens.get("access_token")}
        }
        # Clean URL (remove ?code=...)
        st.experimental_set_query_params()
        st.success(f"Signed in as {info.get('email')}")
    except Exception as e:
        st.error(f"Google sign-in failed: {e}")

def get_current_user():
    return st.session_state.get("user")

def logout_button():
    if st.button("Sign out"):
        st.session_state.pop("user", None)
        st.experimental_set_query_params()

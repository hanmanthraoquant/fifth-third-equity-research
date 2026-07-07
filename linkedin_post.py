"""
LinkedIn posting via the OFFICIAL LinkedIn API (OAuth 2.0 + Posts API).
Standard library only. See LINKEDIN_API_SETUP.md for the one-time app setup.

  python linkedin_post.py --auth                       # get/refresh access token
  python linkedin_post.py --post DRAFT_linkedin_post.txt --dry-run   # preview
  python linkedin_post.py --post DRAFT_linkedin_post.txt             # publish

Credentials are read from environment or a local .env (git-ignored):
  LINKEDIN_CLIENT_ID, LINKEDIN_CLIENT_SECRET, LINKEDIN_REDIRECT_URI,
  LINKEDIN_ACCESS_TOKEN (written by --auth)

This is the ToS-compliant path. Never commit your .env. NOT affiliated with LinkedIn.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

BASE = Path(__file__).parent
ENV = BASE / ".env"
SCOPES = "openid profile email w_member_social"
AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
USERINFO_URL = "https://api.linkedin.com/v2/userinfo"
POSTS_URL = "https://api.linkedin.com/rest/posts"
LINKEDIN_VERSION = "202405"   # YYYYMM; bump if LinkedIn requires a newer version


# ── tiny .env helpers ─────────────────────────────────────────────────────────
def load_env() -> dict:
    d = {}
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("LINKEDIN_CLIENT_ID", "LINKEDIN_CLIENT_SECRET", "LINKEDIN_REDIRECT_URI",
              "LINKEDIN_ACCESS_TOKEN"):
        if os.environ.get(k):
            d[k] = os.environ[k]
    return d


def set_env(key: str, value: str) -> None:
    lines = ENV.read_text(encoding="utf-8", errors="ignore").splitlines() if ENV.exists() else []
    out, found = [], False
    for line in lines:
        if line.strip().startswith(key + "="):
            out.append(f"{key}={value}"); found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    ENV.write_text("\n".join(out) + "\n", encoding="utf-8")


# ── OAuth ─────────────────────────────────────────────────────────────────────
class _Handler(BaseHTTPRequestHandler):
    code = None

    def do_GET(self):
        q = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(q)
        _Handler.code = params.get("code", [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        msg = "Authorization received — you can close this tab and return to the terminal."
        if not _Handler.code:
            msg = "No code received. Error: " + params.get("error_description", ["unknown"])[0]
        self.wfile.write(f"<html><body style='font-family:sans-serif'><h3>{msg}</h3></body></html>".encode())

    def log_message(self, *a):  # silence
        pass


def do_auth(env: dict) -> None:
    cid = env.get("LINKEDIN_CLIENT_ID"); secret = env.get("LINKEDIN_CLIENT_SECRET")
    redirect = env.get("LINKEDIN_REDIRECT_URI", "http://localhost:8000/callback")
    if not (cid and secret):
        raise SystemExit("Set LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET in .env first "
                         "(see LINKEDIN_API_SETUP.md).")
    params = {"response_type": "code", "client_id": cid, "redirect_uri": redirect,
              "scope": SCOPES, "state": "fitb"}
    url = AUTH_URL + "?" + urllib.parse.urlencode(params)
    print("Opening browser to authorize... if it doesn't open, paste this URL:\n" + url)
    webbrowser.open(url)

    host, port = "localhost", int(urllib.parse.urlparse(redirect).port or 8000)
    print(f"Waiting for the redirect on http://{host}:{port} ...")
    srv = HTTPServer((host, port), _Handler)
    srv.handle_request()  # serve exactly one request (the callback)
    code = _Handler.code
    if not code:
        raise SystemExit("No authorization code received. Check the app's redirect URL.")

    body = urllib.parse.urlencode({
        "grant_type": "authorization_code", "code": code, "redirect_uri": redirect,
        "client_id": cid, "client_secret": secret}).encode()
    req = urllib.request.Request(TOKEN_URL, data=body,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=30) as r:
        tok = json.loads(r.read().decode())
    access = tok.get("access_token")
    if not access:
        raise SystemExit("Token exchange failed: " + json.dumps(tok))
    set_env("LINKEDIN_ACCESS_TOKEN", access)
    print(f"Access token saved to .env (expires in ~{tok.get('expires_in', 0)//86400} days).")


# ── posting ───────────────────────────────────────────────────────────────────
def _member_urn(token: str) -> str:
    req = urllib.request.Request(USERINFO_URL, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        info = json.loads(r.read().decode())
    sub = info.get("sub")
    if not sub:
        raise SystemExit("Could not read member id (userinfo). Re-run --auth with openid+profile scope.")
    return f"urn:li:person:{sub}"


def do_post(env: dict, text: str, dry: bool) -> None:
    token = env.get("LINKEDIN_ACCESS_TOKEN")
    if not token:
        raise SystemExit("No LINKEDIN_ACCESS_TOKEN. Run: python linkedin_post.py --auth")
    if dry:
        print("--- DRY RUN (not posting) ---\n" + text + "\n--- end ---")
        return
    author = _member_urn(token)
    payload = {
        "author": author,
        "commentary": text,
        "visibility": "PUBLIC",
        "distribution": {"feedDistribution": "MAIN_FEED", "targetEntities": [],
                         "thirdPartyDistributionChannels": []},
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
    }
    req = urllib.request.Request(
        POSTS_URL, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                 "X-Restli-Protocol-Version": "2.0.0", "LinkedIn-Version": LINKEDIN_VERSION})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            post_id = r.headers.get("x-restli-id") or r.headers.get("x-linkedin-id")
            print(f"Posted to LinkedIn. Post id: {post_id}")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Post failed ({e.code}): {e.read().decode()}")


def main():
    ap = argparse.ArgumentParser(description="Post to LinkedIn via the official API")
    ap.add_argument("--auth", action="store_true", help="run OAuth and save access token")
    ap.add_argument("--post", type=str, metavar="FILE", help="text file to publish")
    ap.add_argument("--dry-run", action="store_true", help="preview, do not publish")
    args = ap.parse_args()

    env = load_env()
    if args.auth:
        do_auth(env)
    elif args.post:
        text = Path(args.post).read_text(encoding="utf-8").strip()
        do_post(env, text, args.dry_run)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

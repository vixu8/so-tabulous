"""One-time helper to mint a Box OAuth 2.0 refresh token for the preprocessing run.

Box access tokens last only 60 min, so unattended runs use the OAuth refresh flow
(preprocess_synthtab_box.py auto-trades the refresh token for fresh access tokens).
This script does the one-time 3-legged authorization: it prints the Box consent URL,
takes the `code` you get back, and exchanges it for a refresh token.

Run it ONCE PER INSTANCE -- Box refresh tokens are single-use and rotate, so two
instances must never share one (they'd invalidate each other). Write each token
straight into that instance's persistence file with --out.

Needs: requests. Client id/secret come from BOX_CLIENT_ID / BOX_CLIENT_SECRET (or
--client-id / --client-secret). Your Box app must use Standard OAuth 2.0 (User
Authentication) and have the redirect URI below registered under its config.

Typical use (on the instance, or anywhere -- the token is portable):
    BOX_CLIENT_ID=... BOX_CLIENT_SECRET=... \
    python scripts/box_oauth_setup.py --out ~/.synthtab_box_refresh
  -> prints a URL; open it in a browser, approve, copy the `code=...` from the
     redirect, paste it back; the refresh token is written to --out.
"""

import argparse
import os
import sys
import urllib.parse

import requests

AUTHORIZE = "https://account.box.com/api/oauth2/authorize"
TOKEN = "https://api.box.com/oauth2/token"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--client-id", default=os.environ.get("BOX_CLIENT_ID"))
    ap.add_argument("--client-secret", default=os.environ.get("BOX_CLIENT_SECRET"))
    ap.add_argument("--redirect-uri", default="https://localhost",
                    help="must EXACTLY match a redirect URI registered on the Box app "
                         "(default https://localhost)")
    ap.add_argument("--code", default=None,
                    help="the authorization code from the redirect; if omitted, the "
                         "script prints the consent URL and reads the code from stdin")
    ap.add_argument("--out", default=None,
                    help="write the refresh token to this file (point it at the same "
                         "path you'll pass preprocess_synthtab_box.py --refresh-token-file, "
                         "default there is ~/.synthtab_box_refresh). If omitted, the "
                         "token is only printed.")
    args = ap.parse_args()
    if not args.client_id or not args.client_secret:
        raise SystemExit("set BOX_CLIENT_ID and BOX_CLIENT_SECRET (or pass "
                         "--client-id/--client-secret)")

    code = args.code
    if not code:
        q = urllib.parse.urlencode({"client_id": args.client_id, "response_type": "code",
                                    "redirect_uri": args.redirect_uri})
        print("1) Open this URL in a browser, log in, and approve:\n")
        print(f"   {AUTHORIZE}?{q}\n")
        print("2) You'll be redirected to "
              f"{args.redirect_uri}/?code=XXXXX (the page itself may not load -- that's\n"
              "   fine, you only need the code from the address bar). It expires in ~30s,\n"
              "   so paste it promptly below.\n")
        code = input("Paste the code here: ").strip()
    if not code:
        raise SystemExit("no code provided")

    r = requests.post(TOKEN, data={"grant_type": "authorization_code", "code": code,
                                   "client_id": args.client_id,
                                   "client_secret": args.client_secret,
                                   "redirect_uri": args.redirect_uri}, timeout=30)
    if r.status_code != 200:
        raise SystemExit(f"token exchange failed ({r.status_code}): {r.text}\n"
                         "Common causes: the code expired (re-run and be quick), or the "
                         "--redirect-uri doesn't match the one registered on the app.")
    tok = r.json()
    refresh = tok["refresh_token"]
    print(f"\naccess_token (expires in {tok.get('expires_in', '?')}s, informational): "
          f"{tok['access_token'][:12]}...")
    if args.out:
        out = os.path.expanduser(args.out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            f.write(refresh)
        print(f"refresh_token written to {out}")
        print("This instance is ready; the preprocessing script will read + rotate it here.")
    else:
        print(f"\nrefresh_token:\n{refresh}\n")
        print("export it for the run:  export BOX_REFRESH_TOKEN='<the token above>'")
    print("\nReminder: this token is single-use/rotating -- mint a SEPARATE one per "
          "instance; never reuse it across two.")


if __name__ == "__main__":
    main()

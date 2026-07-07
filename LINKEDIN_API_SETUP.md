# LinkedIn API integration — setup guide

Goal: get an **official** LinkedIn access token with the `w_member_social` scope so
`linkedin_post.py` can publish a post to your own profile via LinkedIn's API.

> This is the ToS-compliant path (LinkedIn's official OAuth + Posts API). It is the
> only safe way to automate posting — do **not** use cookie/scraper tools, which
> can get your account restricted.

---

## Part A — Register the Developer app (one-time, ~10 min, done by YOU on linkedin.com)

You need a **LinkedIn Company Page** to create an app. If you don't have one:
- LinkedIn → **For Business** (top-right grid) → **Create a Company Page** → pick
  "Small business", fill the basics. (A page for your own project/name is fine.)

Then:
1. Go to **https://www.linkedin.com/developers/apps** and sign in.
2. Click **Create app**.
3. Fill in:
   - **App name**: e.g. `FITB Research Poster`
   - **LinkedIn Page**: select the Company Page from above
   - **App logo**: any image
   - Accept the legal terms → **Create app**.
4. Open the app → **Settings** tab → under "Verify" click **Verify** and follow the
   prompt (a Page admin — you — approves it). This activates the app.

## Part B — Add the products (permissions)

5. Go to the **Products** tab and request:
   - **Sign In with LinkedIn using OpenID Connect** (gives `openid`, `profile`,
     `email` — needed to read your member id). Usually instant.
   - **Share on LinkedIn** (gives `w_member_social` — needed to post). Usually
     self-serve; if it asks for review, follow the prompts.
   > Product names/availability change over time — pick whatever grants
   > `w_member_social`. You can see granted scopes on the **Auth** tab.

## Part C — Get credentials + set the redirect URL

6. Go to the **Auth** tab. Copy:
   - **Client ID**
   - **Client Secret**
7. Under **OAuth 2.0 settings → Authorized redirect URLs**, add exactly:
   ```
   http://localhost:8000/callback
   ```
   Save.

## Part D — Put credentials in your local .env (never commit this)

Add these lines to `FITB_Equity_Research/.env` (already git-ignored):
```
LINKEDIN_CLIENT_ID=your_client_id
LINKEDIN_CLIENT_SECRET=your_client_secret
LINKEDIN_REDIRECT_URI=http://localhost:8000/callback
```

## Part E — Authorize + post

8. Get a token (opens your browser, you click "Allow"):
   ```
   python linkedin_post.py --auth
   ```
   This saves `LINKEDIN_ACCESS_TOKEN` into `.env`. Tokens last ~60 days.
9. Publish the draft:
   ```
   python linkedin_post.py --post DRAFT_linkedin_post.txt
   ```
   (Add `--dry-run` first to preview without posting.)

---

### Notes / honest caveats
- **App review:** if LinkedIn gates `w_member_social` behind review for your app,
  posting won't work until they approve — that's on LinkedIn, not the code.
- **Token expiry:** access tokens expire (~60 days); re-run `--auth` to refresh.
- **What it posts as:** a normal post on *your* member profile (not the Company Page).
- Once you can run step 9 yourself, a future Claude session could run it too — but
  the token/credentials always live in your local `.env`, never in the repo.

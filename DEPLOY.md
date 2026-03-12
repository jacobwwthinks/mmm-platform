# Deploy MMM Platform to Streamlit Cloud

## Step 1: Create a Private GitHub Repo (2 min)

1. Go to **github.com/new**
2. Name it `mmm-platform` (or whatever you prefer)
3. Set to **Private**
4. Don't add README (we already have one)
5. Click **Create repository**

## Step 2: Push the Code (2 min)

Open Terminal and run:

```bash
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/Claude/mmm-platform
git init
git add .
git commit -m "Initial MMM platform"
git branch -M main
git remote add origin https://github.com/jacobwwthinks/mmm-platform.git
git push -u origin main
```

Replace `jacobwwthinks` with your GitHub username.

**Important:** The `.gitignore` ensures `secrets.toml` (with your API key and password) is NOT pushed to GitHub.

## Step 3: Deploy on Streamlit Cloud (3 min)

1. Go to **share.streamlit.io**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your `mmm-platform` repo, `main` branch, and `app.py` as the main file
5. Click **"Advanced settings"** and paste this into the **Secrets** box:

```toml
[windsor]
api_key = "72a03e5596536a32aeb2ce66c478d75d1bd9"

[auth]
password = "Marathon2026!"
```

6. Click **Deploy**

## Step 4: Access Your App

After ~2 minutes, your app will be live at something like:
`https://your-username-mmm-platform-app-xxxxx.streamlit.app`

Share the URL + password with your team. Done!

---

## Updating the App

Any time you push changes to the `main` branch on GitHub, Streamlit Cloud auto-redeploys:

```bash
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/Claude/mmm-platform
git add .
git commit -m "Update description"
git push
```

## Adding a New Client

1. Connect their platforms in Windsor.ai
2. Add their section in `config.yaml` (copy Juniper's format)
3. Push to GitHub — app auto-updates

## Changing the Password

Update it in two places:
- Streamlit Cloud → App settings → Secrets
- (Optional) Your local `.streamlit/secrets.toml`

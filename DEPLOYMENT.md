# Deployment Guide

## Deploying to Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (free tier available at [share.streamlit.io](https://share.streamlit.io))
- Your `.env` variables ready

### Step 1: Push to GitHub

1. Initialize git (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Financial Transcript RAG Assistant"
   ```

2. Create a new repository on GitHub (don't initialize with README)

3. Push to GitHub:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository and branch
5. Set **Main file path** to: `app.py`
6. Click **"Advanced settings"** to add secrets

### Step 3: Add Secrets (Environment Variables)

In Streamlit Cloud, add these secrets in **Settings > Secrets**:

**Important:** Streamlit Cloud uses TOML format, not `.env` format. Use this format:

```toml
OPENAI_API_KEY = "sk-proj-..."
SUPABASE_URL = "https://xxxxx.supabase.co"
SUPABASE_KEY = "sb_publishable_..."
DATABASE_URL = "postgresql://postgres.[ref]:[PASSWORD]@aws-0-[region].pooler.supabase.com:5432/postgres"
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
```

**Note:** 
- Remove the `[secrets]` header - Streamlit Cloud doesn't need it
- All values must be in quotes
- No comments (lines starting with `#`) in the secrets file
- `SUPABASE_SERVICE_KEY` is optional - you can omit it if not needed

**Important:**
- Use the **Session Pooler** connection string for `DATABASE_URL` (not Direct connection)
- Never commit your `.env` file to GitHub
- Streamlit Cloud will automatically load these secrets as environment variables

### Step 4: Deploy

1. Click **"Deploy"**
2. Wait for the app to build and deploy
3. Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

### Troubleshooting

**Build fails:**
- Check that `requirements.txt` has all dependencies
- Verify Python version compatibility (3.11+)

**Database connection errors:**
- Ensure `DATABASE_URL` uses Session Pooler (not Direct connection)
- Check that your Supabase database is accessible from external IPs

**Import errors:**
- Verify all packages in `requirements.txt` are correct
- Check that you're using the latest versions

### Post-Deployment

- Test uploading a PDF
- Test asking questions
- Monitor usage in Streamlit Cloud dashboard
- Check Supabase dashboard for data storage

## Alternative: Deploy to Other Platforms

### Fly.io
- Create `Dockerfile` and `fly.toml`
- Use `flyctl` CLI to deploy
- Set environment variables via `fly secrets set`

### Railway
- Connect GitHub repo
- Set environment variables in dashboard
- Auto-deploys on push

### Heroku
- Use `Procfile` with: `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- Set config vars in dashboard


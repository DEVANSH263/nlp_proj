# 🚀 HateShield Deployment Guide - Railway.app

**Complete step-by-step guide to deploy HateShield to Railway.app with a public URL**

---

## OVERVIEW

- **Platform**: Railway.app
- **Cost**: Free trial → $5/month
- **Setup Time**: 10-15 minutes
- **Final URL**: `https://your-app.railway.app`
- **Models**: All 3 (942 MB - fits!)
- **Database**: SQLite (auto-persisted)

---

## PREREQUISITE CHECKLIST

- [ ] GitHub account (free at github.com)
- [ ] Railway account (free at railway.app)
- [ ] Git installed locally
- [ ] Your project in Git repository

---

## STEP 1: Push Project to GitHub

### 1.1 Create GitHub Repository
```
1. Go to github.com
2. Click "New repository"
3. Name: nlp_proj
4. Add .gitignore (Python)
5. Create repository
```

### 1.2 Upload Your Project
```bash
# In your project folder:
git init
git add .
git commit -m "Initial commit: HateShield with 3 models"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/nlp_proj.git
git push -u origin main
```

**Verify:** Visit `https://github.com/YOUR-USERNAME/nlp_proj` - you should see all files

---

## STEP 2: Prepare Project for Railway

### 2.1 Create `requirements.txt`
```bash
cd c:\Documents\Projects\nlp_proj
pip freeze > requirements.txt
```

**Check file includes:**
```
Flask==2.x.x
Flask-Login==0.6.x
Flask-SQLAlchemy==3.x.x
torch==2.x.x
transformers==4.x.x
scikit-learn==1.x.x
pandas==2.x.x
numpy==2.x.x
gunicorn==21.x.x  # Important for Railway
```

**If missing gunicorn, add:**
```bash
echo "gunicorn==21.2.0" >> requirements.txt
```

### 2.2 Create `Procfile`
**File:** `Procfile` (NO EXTENSION)
```
web: gunicorn -w 2 -b 0.0.0.0:$PORT wsgi:app
```

### 2.3 Create `wsgi.py`
**File:** `wsgi.py` (in root)
```python
import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

### 2.4 Update `config.py`
**Modify** `c:\Documents\Projects\nlp_proj\config.py`:

```python
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Railway provides PORT env var
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Use environment variable for database (Railway auto-provides)
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'sqlite:///' + os.path.join(BASE_DIR, 'database.db')
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Model paths (should work as-is)
    MODEL_PATH = os.path.join(BASE_DIR, 'model3', 'lr', 'model.pkl')
    VECTORIZER_PATH = os.path.join(BASE_DIR, 'model3', 'lr', 'vectorizer.pkl')
    LSTM_MODEL_PATH  = os.path.join(BASE_DIR, 'model3', 'lstm', 'lstm_model.pt')
    LSTM_VOCAB_PATH  = os.path.join(BASE_DIR, 'model3', 'lstm', 'lstm_vocab.pkl')
    MURIL_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'muril')
```

### 2.5 Create `.gitignore`
**File:** `.gitignore` (in root)
```
__pycache__/
*.pyc
*.pyo
.env
venv/
env/
*.db
database.db
.DS_Store
*.egg-info/
dist/
build/
```

### 2.6 Commit Changes
```bash
git add requirements.txt Procfile wsgi.py .gitignore
git commit -m "Add deployment files for Railway"
git push origin main
```

---

## STEP 3: Deploy to Railway

### 3.1 Create Railway Account
```
1. Go to railway.app
2. Click "Login/Sign Up"
3. Use GitHub (easiest) OR email
4. Authorize Railway to access GitHub
```

### 3.2 Create New Project
```
1. Dashboard → Click "New Project"
2. Select "Deploy from GitHub repo"
3. Click "Configure GitHub App"
4. Select your repository: nlp_proj
5. Click "Deploy"
```

### 3.3 Configure Variables (If Needed)
```
In Railway Dashboard:
1. Click your project
2. Go to "Variables" tab
3. Add if needed:
   SECRET_KEY = your-random-secret-key
```

### 3.4 Wait for Build
```
Railway will:
1. Build environment (1-2 min)
2. Install dependencies (3-5 min)
3. Start server (1 min)
4. Total: ~10 minutes

Watch logs for progress
```

---

## STEP 4: Get Your Public URL

### 4.1 Access Dashboard
```
1. Go to railway.app dashboard
2. Click your project: nlp_proj
3. Look for "Deployments" section
4. Find "SERVICE" named "web"
```

### 4.2 Get URL
```
In the "web" service card, look for:
🌐 Domains: https://nlp-proj-production-xxxx.railway.app

COPY THIS URL! This is your public website.
```

### 4.3 Test It Works
```
Open browser:
https://nlp-proj-production-xxxx.railway.app/home

You should see HateShield homepage!
```

---

## STEP 5: Verify All Features

### 5.1 Test Dashboard
```
1. Go to /dashboard
2. Login (or register new account)
3. Test text: "you are stupid"
4. Select model: MuRIL
5. Click "Analyse Now"
6. Should show: HOF (0.807 confidence)
```

### 5.2 Test Compare
```
1. Go to /compare
2. Enter same text
3. Should show all 3 models predictions
```

### 5.3 Test History
```
1. Go to /history
2. Should see past predictions
```

---

## TROUBLESHOOTING

### Problem: "Build Failed"
```
Solution:
1. Check Railway logs (red X icon)
2. Common issue: Missing dependencies
3. Fix: pip freeze > requirements.txt
4. Push to GitHub
5. Railway auto-redeploys
```

### Problem: "Models not found" (500 error)
```
Solution:
1. Ensure model/ and model3/ folders are in GitHub
2. Check they weren't gitignored
3. Verify file paths in config.py
4. Add to .gitignore if too large:
   # .gitignore
   model/muril/pytorch_model.bin  # if > 512MB
5. Use Git LFS for large files
```

### Problem: "Port error" or "Connection refused"
```
Solution:
1. Check wsgi.py has: port = int(os.environ.get('PORT', 5000))
2. Check Procfile: web: gunicorn -w 2 -b 0.0.0.0:$PORT wsgi:app
3. Restart deployment in Railway
```

### Problem: "Database errors"
```
Solution:
1. Railway auto-creates /tmp/database.db
2. If persisting data is needed, use PostgreSQL addon:
   - Railway Dashboard → Add Services → PostgreSQL
   - Auto-sets DATABASE_URL
   - config.py will use it automatically
```

---

## FILE CHECKLIST

Before deployment, ensure these files exist in GitHub:

```
nlp_proj/
├── app.py                    ✅
├── config.py                 ✅ (UPDATED for Railway)
├── wsgi.py                   ✅ (NEW)
├── Procfile                  ✅ (NEW)
├── requirements.txt          ✅ (NEW)
├── .gitignore               ✅ (NEW)
├── models.py                ✅
├── database.db              ✅
├── utils/
│   ├── predict.py          ✅
│   ├── prep2.py            ✅
│   └── normalize.py        ✅
├── routes/
│   ├── main.py             ✅
│   ├── auth.py             ✅
│   └── report.py           ✅
├── templates/              ✅
│   ├── base.html
│   ├── dashboard.html
│   └── ...
├── static/                 ✅
│   ├── css/
│   ├── js/
├── model3/                 ✅ (942 MB)
│   ├── lr/
│   └── lstm/
└── model/                  ✅ (915 MB MuRIL)
    └── muril/
```

---

## DEPLOYMENT CHECKLIST

- [ ] Project pushed to GitHub
- [ ] requirements.txt created & includes gunicorn
- [ ] Procfile created
- [ ] wsgi.py created
- [ ] config.py updated for Railway
- [ ] .gitignore created
- [ ] Railway account created
- [ ] Project deployed from GitHub
- [ ] URL obtained: https://xxx.railway.app
- [ ] Dashboard tested
- [ ] Compare tested
- [ ] History tested

---

## FINAL URLS

After deployment, you'll have:

```
🏠 Home:       https://xxx.railway.app/home
📊 Dashboard:  https://xxx.railway.app/dashboard
🔄 Compare:    https://xxx.railway.app/compare
📜 History:    https://xxx.railway.app/history
```

**SHARE THIS URL WITH USERS!** ✅

---

## NEXT STEPS (OPTIONAL)

### Add Custom Domain
```
1. Buy domain (godaddy.com, namecheap.com) ~$10/year
2. Railway Dashboard → Settings → Domains
3. Add your domain
4. Point nameservers to Railway (instructions provided)
5. Done! Use: https://yourdomain.com
```

### Enable Database Backups
```
1. Railway → PostgreSQL Add-on
2. Auto-backups every day
3. Restore from any backup
```

### Monitor Performance
```
1. Railway → Metrics tab
2. View CPU, Memory, Network usage
3. Get alerts if usage spikes
```

---

## IF CHAT LIMIT REACHED

**You have everything needed to deploy!**

1. **Follow steps 1-5 exactly** as written above
2. **Troubleshooting section** covers 80% of issues
3. **Railway support** available at railway.app/help
4. **Your models work locally** - they'll work on Railway

**Most common reason for failure:** Models not in GitHub (too large)
- **Solution:** Use Git LFS or upload separately

---

## SUPPORT RESOURCES

- Railway docs: https://docs.railway.app
- Git help: https://docs.github.com/en
- Flask deployment: https://flask.palletsprojects.com/deployment/
- Gunicorn docs: https://docs.gunicorn.org/

---

**YOU'VE GOT THIS! 🚀**

Questions not covered? Check the Railway docs or contact their support.

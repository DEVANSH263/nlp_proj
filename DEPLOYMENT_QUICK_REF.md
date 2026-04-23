# 🚀 Railway.app Deployment - Quick Reference

## BACKUP GUIDE (If chat limit reached)

This file contains everything needed to deploy HateShield to Railway.app

---

## FILES CREATED FOR DEPLOYMENT

✅ **Procfile** - Tells Railway how to run the app
✅ **wsgi.py** - WSGI entry point for Gunicorn
✅ **requirements.txt** - Python dependencies
✅ **.gitignore** - What NOT to commit to Git
✅ **RAILWAY_DEPLOYMENT_GUIDE.md** - Full step-by-step guide

---

## QUICK START (5 STEPS)

### Step 1: Initialize Git
```bash
cd c:\Documents\Projects\nlp_proj
git init
git add .
git commit -m "Initial commit"
```

### Step 2: Create GitHub repo and push
```bash
# Go to github.com, create "nlp_proj" repo
git remote add origin https://github.com/YOUR-USERNAME/nlp_proj.git
git branch -M main
git push -u origin main
```

### Step 3: Create Railway account
```
Go to railway.app
Sign up with GitHub
```

### Step 4: Deploy
```
Railway dashboard → New Project → Deploy from GitHub
Select: nlp_proj → Deploy
Wait 10 minutes for build
```

### Step 5: Get URL
```
Railway dashboard → project → web service
Copy: https://xxx.railway.app
Open in browser!
```

---

## KEY CONFIGURATION

**config.py** - Already configured for Railway:
```python
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'  # Auto-persisted in /tmp/
MODEL_PATH = 'model3/lr/model.pkl'
LSTM_MODEL_PATH = 'model3/lstm/lstm_model.pt'
MURIL_MODEL_PATH = 'model/muril'
```

**Procfile** - Railway entrypoint:
```
web: gunicorn -w 2 -b 0.0.0.0:$PORT wsgi:app
```

**wsgi.py** - Flask app wrapper:
```python
from app import create_app
app = create_app()
```

---

## MODEL SIZES

- LR: 7.33 MB
- LSTM: 18.83 MB
- MuRIL: 915.41 MB
- **Total: 942 MB** ✅ (Fits on Railway)

---

## DEPLOYMENT CHECKLIST

- [ ] GitHub repo created
- [ ] Code pushed to main branch
- [ ] requirements.txt present
- [ ] Procfile present
- [ ] wsgi.py present
- [ ] Railway account created
- [ ] Project deployed
- [ ] URL obtained
- [ ] Dashboard works
- [ ] Compare works
- [ ] History works

---

## TROUBLESHOOTING

**Build fails?**
→ Check Railway logs (red icon)
→ Usually missing dependency
→ Run: `pip freeze > requirements.txt`
→ Push to GitHub, Railway auto-redeploys

**Models not found?**
→ Ensure model/ and model3/ folders in GitHub
→ Not gitignored
→ Check Railway logs for 500 errors

**URL doesn't work?**
→ Wait 5 minutes for build to complete
→ Check deployment status: green = ready
→ Try clearing browser cache

**Database issues?**
→ Railway auto-creates /tmp/database.db
→ Persists between deployments
→ Optional: Add PostgreSQL addon for production

---

## FINAL URLS

After successful deployment:

```
🏠 Home:      https://xxx.railway.app/home
📊 Dashboard: https://xxx.railway.app/dashboard
🔄 Compare:   https://xxx.railway.app/compare
📜 History:   https://xxx.railway.app/history
```

---

## IF STUCK

1. **Read:** RAILWAY_DEPLOYMENT_GUIDE.md (full guide)
2. **Search:** Railway docs (docs.railway.app)
3. **Check:** Railway logs in dashboard
4. **Contact:** Railway support chat

---

## NEXT STEPS (OPTIONAL)

- Add custom domain (~$10/year)
- Enable PostgreSQL for production
- Setup monitoring
- Configure backups

---

**DEPLOYMENT IS STRAIGHTFORWARD!**

✅ Config ready
✅ Files ready
✅ Models ready
✅ Guide ready

Just push to GitHub and deploy! 🚀

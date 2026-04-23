# 🚀 SCENARIO 3 DEPLOYMENT GUIDE (Final)

**Deploy: LR + LSTM from model3/ → Railway | MuRIL from model/ → HuggingFace**

---

## 📋 What Gets Deployed

| Model | Source | Deploy To | Size |
|-------|--------|-----------|------|
| **LR** | `model3/lr/model.pkl` | Railway | 5.5 MB |
| **Vectorizer** | `model3/lr/vectorizer.pkl` | Railway | 1.8 MB |
| **LSTM** | `model3/lstm/lstm_model.pt` | Railway | 18.83 MB |
| **MuRIL** | `model/muril/` | HuggingFace | 915 MB |
| **TOTAL** | - | **26 MB Railway + 915 MB HF** | ✅ |

---

## 🎯 Step-by-Step (30 minutes)

### PHASE 1: Upload MuRIL to HuggingFace (5 min)

#### 1.1 Create HuggingFace Account
```
1. Go to: https://huggingface.co/join
2. Sign up (free)
3. Verify email
```

#### 1.2 Get Your Token
```
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "railway-deployment"
4. Permissions: write
5. Create
6. Copy token (save somewhere safe)
```

#### 1.3 Upload MuRIL Model
```bash
cd c:\Documents\Projects\nlp_proj

# Install HuggingFace CLI (one-time)
pip install huggingface-hub

# Login
huggingface-cli login
# Paste your token when prompted

# Run upload script
python UPLOAD_TO_HUGGINGFACE.py

# Follow prompts:
# - Enter HuggingFace username
# - Wait for upload (5-10 min)
```

#### 1.4 Verify Upload
```
Go to: https://huggingface.co/YOUR-USERNAME/hateshield-muril
Should see:
  ✅ pytorch_model.bin
  ✅ config.json
  ✅ tokenizer.json
  ✅ vocab.txt
  ✅ special_tokens_map.json
```

---

### PHASE 2: Update Code for Railway (5 min)

#### 2.1 Update config.py

Change:
```python
# config.py line 15
MURIL_MODEL_PATH = os.environ.get('MURIL_MODEL_ID', 'USERNAME/hateshield-muril')
```

To:
```python
# config.py line 15
MURIL_MODEL_PATH = os.environ.get('MURIL_MODEL_ID', 'YOUR-HUGGINGFACE-USERNAME/hateshield-muril')
```

Replace `YOUR-HUGGINGFACE-USERNAME` with your actual username!

#### 2.2 Verify .gitignore

Should have:
```
model/              # ✅ ignored (use HF)
model2/             # ✅ ignored (old)
model3/muril/       # ✅ ignored (use HF)
# model3/lr/        # NOT ignored (included)
# model3/lstm/      # NOT ignored (included)
```

#### 2.3 Check requirements.txt

Make sure includes:
```
Flask==2.3.3
Flask-SQLAlchemy==3.0.5
torch==2.5.1
transformers==4.44.2
scikit-learn==1.7.1
gunicorn==21.2.0
```

---

### PHASE 3: Push to GitHub (5 min)

#### 3.1 Initialize Git
```bash
cd c:\Documents\Projects\nlp_proj
git init
```

#### 3.2 Create GitHub Repository
```
1. Go to: https://github.com/new
2. Name: nlp_proj
3. Description: "HateShield - 3 Model Hate Speech Detection"
4. Public (so Railway can access)
5. Create repository
```

#### 3.3 Add Remote & Push
```bash
# Copy from GitHub (replace YOUR-USERNAME)
git remote add origin https://github.com/YOUR-USERNAME/nlp_proj.git
git branch -M main
git add .
git commit -m "Scenario 3: LR+LSTM Railway, MuRIL HuggingFace"
git push -u origin main

# Wait for completion (~2 min for 26 MB)
```

#### 3.4 Verify GitHub
```
Visit: https://github.com/YOUR-USERNAME/nlp_proj
Should see:
  ✅ app.py
  ✅ routes/
  ✅ model3/lr/ (with model.pkl + vectorizer.pkl)
  ✅ model3/lstm/ (with lstm_model.pt)
  ✅ Procfile
  ✅ wsgi.py
  ✅ requirements.txt
```

---

### PHASE 4: Deploy to Railway (10 min)

#### 4.1 Create Railway Account
```
1. Go to: https://railway.app
2. Sign up (use GitHub for easy login)
3. Authorize Railway
```

#### 4.2 Deploy Project
```
1. Railway Dashboard → "New Project"
2. Click "Deploy from GitHub repo"
3. Select: YOUR-USERNAME/nlp_proj
4. Click "Deploy"
5. Wait for build (watch logs)
```

#### 4.3 Add Environment Variable (CRITICAL)
```
Railway Dashboard → Your Project → Variables

Add:
  KEY:   MURIL_MODEL_ID
  VALUE: YOUR-USERNAME/hateshield-muril

Save
```

#### 4.4 Wait for Build
```
Railway logs show:
  Installing dependencies...      (3 min)
  Collecting packages...          (2 min)
  Building...                     (1 min)
  Starting server...              (1 min)
  
Total: ~7-10 minutes

✅ When done: "Deployment successful"
```

---

### PHASE 5: Test Deployment (5 min)

#### 5.1 Get Public URL
```
Railway Dashboard → Your Project → Deployments
Look for "web" service
Copy: https://your-app-xxxx.railway.app
```

#### 5.2 Test Homepage
```
Browser: https://your-app-xxxx.railway.app/home
Should see: HateShield homepage ✅
```

#### 5.3 Test LR Model (instant)
```
1. Go to /dashboard
2. Register or login
3. Text: "you are stupid"
4. Model: LR
5. Predict
Expected: HOF (81% confidence) ✅
```

#### 5.4 Test LSTM Model (instant)
```
1. Go to /dashboard
2. Same text
3. Model: LSTM
4. Predict
Expected: HOF (80% confidence) ✅
```

#### 5.5 Test MuRIL Model (first request: 2-3 min)
```
1. Go to /dashboard
2. Same text
3. Model: MuRIL
4. Click "Predict"

FIRST request:
  ⏳ Downloading from HuggingFace...
  ⏳ Loading model...
  ⏳ Predicting... (takes 2-3 min)
  Result: HOF (83% confidence) ✅

SUBSEQUENT requests:
  ✅ Model cached
  ✅ Instant (0.5-2 sec) ✅
```

#### 5.6 Test Compare Page
```
1. Go to /compare
2. Text: "tu bilkul pagal hai" (Hinglish)
3. Click "Compare All Models"
4. Should see all 3:
  - LR:    NOT/HOF + confidence
  - LSTM:  NOT/HOF + confidence
  - MuRIL: NOT/HOF + confidence ✅
```

---

## ✅ Deployment Checklist

- [ ] MuRIL uploaded to HuggingFace
- [ ] HuggingFace model visible at https://hf.co/YOUR-USERNAME/hateshield-muril
- [ ] config.py updated with your HF username
- [ ] .gitignore excludes model/, model2/, model3/muril/
- [ ] model3/lr/ + model3/lstm/ included in git
- [ ] requirements.txt has gunicorn
- [ ] Procfile exists
- [ ] wsgi.py exists
- [ ] GitHub repo created
- [ ] Project pushed to GitHub
- [ ] Railway project deployed
- [ ] MURIL_MODEL_ID environment variable set
- [ ] LR model works (instant)
- [ ] LSTM model works (instant)
- [ ] MuRIL model works (2-3 min first, then instant)
- [ ] Compare page shows all 3 models

---

## 🎯 What Happens After Deploy

### First Request (MuRIL):
```
Browser → Railway
         ↓
         Downloads MuRIL from HF (2-3 min)
         ↓
         Loads into memory
         ↓
         Returns prediction
         ↓
Browser shows: ✅ HOF 0.83 (after waiting)
```

### Subsequent Requests:
```
Browser → Railway
         ↓
         MuRIL already cached
         ↓
         Returns prediction instantly
         ↓
Browser shows: ✅ HOF 0.83 (0.5-2 sec)
```

---

## ⚠️ Troubleshooting

### Problem: "Build timeout"
```
Solution:
  1. Check GitHub - repo should be ~26 MB (not 942 MB)
  2. Verify model3/muril/ is gitignored
  3. Restart deployment in Railway
```

### Problem: "MuRIL not loading" (500 error)
```
Solution:
  1. Check Railway logs: https://xxx.railway.app
  2. Verify MURIL_MODEL_ID variable set correctly
  3. Check HF model exists: https://hf.co/YOUR-USERNAME/hateshield-muril
  4. Re-set environment variable in Railway
  5. Restart deployment
```

### Problem: "LR or LSTM not loading"
```
Solution:
  1. Verify model3/lr/ in GitHub repo
  2. Verify model3/lstm/ in GitHub repo
  3. Check paths in config.py
  4. Restart Railway deployment
```

### Problem: "Takes too long for first MuRIL request"
```
This is NORMAL:
  First request: 2-3 minutes (downloading 915 MB from HF)
  Subsequent: 0.5-2 sec (cached)
  
Solution: Wait it out, then refresh page when done
```

---

## 📊 Final Summary

| Component | Status | Location |
|-----------|--------|----------|
| **LR Model** | ✅ GitHub + Railway | model3/lr/ |
| **LSTM Model** | ✅ GitHub + Railway | model3/lstm/ |
| **MuRIL Model** | ✅ HuggingFace Hub | huggingface.co/USERNAME/hateshield-muril |
| **Deployment** | ✅ Railway.app | your-app.railway.app |
| **Cost** | ✅ Free | $0/month (free tier) |

---

## 🎉 Success Criteria

Your deployment works when:
```
✅ https://your-app.railway.app/home loads
✅ /dashboard: LR predicts (instant)
✅ /dashboard: LSTM predicts (instant)
✅ /dashboard: MuRIL predicts (2-3 min first time, then instant)
✅ /compare: All 3 models show
✅ Database saves predictions
✅ User login/register works
```

---

**YOU'RE READY TO DEPLOY! 🚀**

Questions? Check:
- DEPLOYMENT_SCENARIOS.md (overview)
- Railway docs: https://docs.railway.app
- HF docs: https://huggingface.co/docs

#  HateShield - Hate Speech Detection System

**Multi-Model Hate Speech Detection for English & Hindi Content**

![Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![Models](https://img.shields.io/badge/Models-3-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-82%25-brightgreen)

---

## Overview

HateShield is a Flask-based web application that detects hate speech and offensive content in English and Hindi using three complementary machine learning models.

**Try it live:** (Add your Railway URL here)

---

## Features

✅ **3 ML Models**
- Logistic Regression (TF-IDF) - Fast, interpretable
- BiLSTM Neural Network - Good balance
- MuRIL Transformer - Best accuracy (83%)

✅ **Multi-Language Support**
- English hate speech detection
- Hindi/Hinglish content analysis
- Mixed language support

✅ **User Features**
- Dashboard: Single model prediction
- Compare: Side-by-side all 3 models
- History: Track past predictions
- User authentication & persistence

✅ **Model-Specific Preprocessing**
- LR: Aggressive cleaning (URLs, emojis, stopwords)
- LSTM: Moderate cleaning (preserves word order)
- MuRIL: Minimal cleaning (transformer handles complexity)

---

## Models Performance

| Model | Type | Accuracy | Size | Speed |
|-------|------|----------|------|-------|
| **LR** | TF-IDF + Logistic Regression | 81.61% | 7.33 MB | ⚡ Fast |
| **LSTM** | BiLSTM with Attention | 80.47% | 18.83 MB | Medium |
| **MuRIL** | Transformer (237M params) | 83.46% | 915 MB | Slower |

---

## Project Structure

```
nlp_proj/
├── app.py                    # Flask app initialization
├── models.py                 # SQLAlchemy ORM models
├── config.py                 # Configuration
├── wsgi.py                   # WSGI entry (production)
├── Procfile                  # Railway deployment
├── requirements.txt          # Python dependencies
│
├── utils/
│   ├── predict.py           # Unified prediction interface
│   ├── prep2.py             # Model-specific preprocessing
│   └── normalize.py         # Hinglish normalization
│
├── routes/
│   ├── main.py              # Dashboard, Compare, History
│   ├── auth.py              # User authentication
│   └── report.py            # Report generation
│
├── templates/               # HTML templates
│   ├── base.html
│   ├── dashboard.html
│   ├── compare.html
│   ├── history.html
│   └── ...
│
├── static/                  # CSS, JS
│   ├── css/style.css
│   └── js/main.js
│
├── model3/                  # Trained models (latest)
│   ├── lr/
│   ├── lstm/
│   └── muril/
│
├── model/                   # Original models
│   └── muril/
│
└── datasets/                # Training data
    ├── english_dataset/
    ├── hindi_dataset/
    └── ...
```

---

## Quick Start (Local)

### 1. Clone Repository
```bash
git clone https://github.com/YOUR-USERNAME/nlp_proj.git
cd nlp_proj
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Development Server
```bash
python app.py
```

Open browser: `http://127.0.0.1:5000`

---

## Deployment (Railway.app)

See [RAILWAY_DEPLOYMENT_GUIDE.md](RAILWAY_DEPLOYMENT_GUIDE.md) for complete instructions.

**Quick deploy:**
```bash
1. Push to GitHub
2. Railway.app → Deploy from GitHub
3. Get public URL (5-10 minutes)
4. Done!
```

**Cost:** Free trial → $5/month

---

## Usage Examples

### Web Interface
```
1. Go to /dashboard
2. Enter text: "you are stupid"
3. Select model: MuRIL
4. Click "Analyse Now"
5. See prediction: HOF (0.807 confidence)
```

### API Endpoint (Future)
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "you are stupid", "model": "muril"}'

Response:
{
  "prediction": "HOF",
  "confidence": 0.807,
  "model_used": "MuRIL"
}
```

---

## Dataset

**HASOC 2019 + 2020**
- Total samples: 17,188 (training)
- Test samples: 3,948
- Languages: English, Hindi
- Classes: HOF (Hate/Offensive), NOT (Safe)
- Class balance: 1.31:1 ratio

---

## Training & Evaluation

**Training Scripts** in `train3/`:
```bash
# Train LR
python train3/train_model.py

# Train LSTM
python train3/train_lstm.py

# Train MuRIL
python train3/train_muril.py
```

**Test Results** from `test_comprehensive.py`:
- LR: 76.5% accuracy on diverse samples
- LSTM: 70.6% accuracy
- MuRIL: 76.5% accuracy (best on Hinglish)

---

## Configuration

**Environment Variables:**
```bash
FLASK_ENV=production
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://...  # Optional
```

**Model Paths** (config.py):
```python
MODEL_PATH = 'model3/lr/model.pkl'
LSTM_MODEL_PATH = 'model3/lstm/lstm_model.pt'
MURIL_MODEL_PATH = 'model/muril/'
```

---

## API Routes

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirect to home |
| `/home` | GET | Landing page |
| `/dashboard` | GET/POST | Single model prediction |
| `/compare` | GET/POST | All 3 models comparison |
| `/history` | GET | User prediction history |
| `/auth/register` | GET/POST | User registration |
| `/auth/login` | GET/POST | User login |
| `/auth/logout` | GET | User logout |

---

## Technologies

**Backend:**
- Flask 2.3.3
- SQLAlchemy 2.0
- PyTorch 2.5.1
- Transformers 4.44.2
- scikit-learn 1.7.1

**Frontend:**
- HTML5 / CSS3
- JavaScript
- Bootstrap (optional)

**Deployment:**
- Railway.app
- Gunicorn
- Docker (optional)

---

## Model Details

### Logistic Regression (LR)
```
Pipeline: TF-IDF → Logistic Regression
Max Features: 5000
Min DF: 2, Max DF: 0.8
Size: 7.33 MB
Speed: Fast
Accuracy: 81.61%
```

### BiLSTM
```
Architecture: Embedding → BiLSTM → Attention → FC
Embedding Dim: 100
Hidden Dim: 128
Num Layers: 2
Dropout: 0.2
Size: 18.83 MB
Speed: Medium
Accuracy: 80.47%
```

### MuRIL
```
Base: google/muril-base-cased (237M params)
Max Length: 128
Batch Size: 16
Learning Rate: 2e-5
Size: 915 MB
Speed: Slower (transformer)
Accuracy: 83.46%
```

---

## Results

**Test Accuracy (Comprehensive Test - 17 samples):**
```
✓ LR:    76.5% (13/17)
✓ LSTM:  70.6% (12/17)
✓ MuRIL: 76.5% (13/17)
```

**Common Failures:**
- Generic phrases ("I hate you") - models trained on contextual hate
- Very short inputs - insufficient context
- Mixed code-switching - moderate support

---

## Known Limitations

- Requires 1+ GB RAM for MuRIL inference
- Transformer models slower (3-5 sec per prediction)
- Limited support for very short texts
- Database limited (SQLite) - use PostgreSQL for scale
- No batch API (yet)

---

## Future Enhancements

- [ ] REST API for batch predictions
- [ ] Real-time stream processing
- [ ] Model explanation (attention visualization)
- [ ] Additional languages (Urdu, Bengali, Tamil)
- [ ] Model quantization (reduce size 75%)
- [ ] PostgreSQL database
- [ ] Redis caching
- [ ] Docker containerization

---

## Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/xyz`)
3. Commit changes (`git commit -am 'Add xyz'`)
4. Push to branch (`git push origin feature/xyz`)
5. Create Pull Request

---

## License

MIT License - See LICENSE file for details

---

## Author

Created by: [Your Name]
Date: April 2026
Version: 1.0.0

---

## Acknowledgments

- HASOC dataset (Indian Language Hate Speech Detection)
- HuggingFace Transformers
- PyTorch community
- Railway.app for hosting

---

## Support

**Issues:** Create GitHub issue
**Discussions:** GitHub Discussions
**Contact:** email@example.com

---

## Deployment Status

- ✅ Local: `http://127.0.0.1:5000`
- 🚀 Production: `https://xxx.railway.app`

**Live Demo:** (Add link after deployment)

---

**Built with ❤️ for hate speech detection**

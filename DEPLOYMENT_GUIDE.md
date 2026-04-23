# HateShield Deployment Approach

## Current Configuration (Production-Ready)

```
Models:
├── LR:    model3/lr/          (81.61% accuracy, TF-IDF)
├── LSTM:  model3/lstm/        (80.47% accuracy, BiLSTM)
└── MuRIL: model/muril/        (83.46% accuracy, best)

Database: SQLite (database.db)
Routes: 12 active endpoints
Users: SQLAlchemy ORM with authentication
```

---

## Deployment Strategy (3 Options)

### OPTION 1: LOCAL DEVELOPMENT (Current)
**Best for**: Testing, demos, small team

**Steps:**
```
1. conda activate test2
2. cd c:\Documents\Projects\nlp_proj
3. python app.py
4. Open http://127.0.0.1:5000
```

**Pros:** Fast, no setup
**Cons:** Single-threaded, not scalable, debug mode on

---

### OPTION 2: PRODUCTION SERVER (Recommended for deployment)
**Best for**: Public deployment, scaling, 24/7 uptime

**Environment:** Windows Server / Ubuntu Server

#### Step 1: Install Requirements
```bash
pip install gunicorn
pip install python-dotenv
```

#### Step 2: Update Config
Create `.env` file:
```
FLASK_ENV=production
SECRET_KEY=your-secure-random-key-here
DATABASE_URL=sqlite:///production.db
```

#### Step 3: Create WSGI Entry Point
Create `wsgi.py`:
```python
import os
from app import create_app
from dotenv import load_dotenv

load_dotenv()
app = create_app()

if __name__ == "__main__":
    app.run()
```

#### Step 4: Run Production Server
```bash
# With 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 wsgi:app

# Or with uWSGI (better performance)
pip install uwsgi
uwsgi --http :5000 --wsgi-file wsgi.py --callable app --processes 4 --threads 2
```

#### Step 5: Use Reverse Proxy (Nginx)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }
}
```

---

### OPTION 3: CLOUD DEPLOYMENT (Heroku, AWS, Azure)

#### Heroku Example:
```bash
# Install Heroku CLI
heroku login
heroku create your-app-name

# Add Procfile
echo "web: gunicorn -w 4 wsgi:app" > Procfile

# Add requirements.txt
pip freeze > requirements.txt

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

---

## Production Checklist

### Security
- [ ] Change `SECRET_KEY` (never use dev key in production)
- [ ] Enable HTTPS (SSL certificate)
- [ ] Use PostgreSQL instead of SQLite
- [ ] Add rate limiting
- [ ] Set strong admin password

### Performance
- [ ] Use Gunicorn/uWSGI (4-8 workers)
- [ ] Add Redis caching for model predictions
- [ ] Compress models (quantization)
- [ ] Enable gzip compression

### Monitoring
- [ ] Add logging (ELK stack)
- [ ] Monitor server health (Prometheus)
- [ ] Set up alerts for errors
- [ ] Track model performance

### Scaling
- [ ] Load balance across multiple servers
- [ ] Use separate model server (async queue)
- [ ] Containerize with Docker
- [ ] Use Kubernetes for orchestration

---

## Recommended: Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_ENV=production
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "wsgi:app"]
```

### docker-compose.yml
```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - ./database.db:/app/database.db
    restart: always
```

### Deploy with Docker
```bash
docker build -t hateshield:latest .
docker run -p 5000:5000 hateshield:latest
```

---

## Model Deployment Strategy

### Current Models (82% avg accuracy)
```
model/muril/          → Best overall (83.46%)
model3/lstm/          → Good for formal language (80.47%)
model3/lr/            → Fast, interpretable (81.61%)
```

### Optimization for Production
1. **Model Quantization**: Reduce size by 75%
   ```python
   # Convert MuRIL to ONNX for faster inference
   from transformers import convert_graph_to_onnx
   ```

2. **Model Caching**: Redis for predictions
   ```python
   # Cache predictions for 1 hour
   cache.set(f"pred_{hash(text)}", result, timeout=3600)
   ```

3. **Async Processing**: For batch predictions
   ```python
   from celery import Celery
   # Process heavy predictions in background
   ```

---

## Database Migration (SQLite → PostgreSQL)

For production, use PostgreSQL:

```python
# Update config.py
SQLALCHEMY_DATABASE_URI = 'postgresql://user:pass@localhost/hateshield'
```

```bash
# Install driver
pip install psycopg2-binary

# Create database
createdb hateshield

# Migrate data
flask db upgrade
```

---

## Deployment Timeline

| Phase | Duration | Action |
|-------|----------|--------|
| **Phase 1** | Day 1 | Setup server, install dependencies |
| **Phase 2** | Day 1-2 | Configure Nginx, SSL certificate |
| **Phase 3** | Day 2 | Deploy Flask app, test endpoints |
| **Phase 4** | Day 3 | Monitor, optimize, scale |
| **Phase 5** | Week 2+ | Production support, updates |

---

## Final Production Stack

```
┌─────────────────────────────────────────┐
│          User (Browser)                 │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      Nginx (Reverse Proxy)              │
│      - SSL/HTTPS                        │
│      - Load balancing                   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│    Gunicorn (4 workers)                 │
│    - Flask App (app.py)                 │
│    - Model Serving                      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│    PostgreSQL Database                  │
│    - User accounts                      │
│    - Prediction history                 │
└─────────────────────────────────────────┘
        │                    │
        └────────┬───────────┘
                 │
        ┌────────▼────────┐
        │   Redis Cache   │
        │ - Model cache   │
        │ - Session store │
        └─────────────────┘
```

---

## Quick Start Commands

### Local Dev (Current)
```bash
conda activate test2
cd c:\Documents\Projects\nlp_proj
python app.py
```

### Production (Recommended)
```bash
# Setup
pip install gunicorn psycopg2-binary python-dotenv

# Configure
echo "FLASK_ENV=production\nSECRET_KEY=<random-key>" > .env

# Create wsgi.py (see above)

# Run
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
```

### Docker (Easiest)
```bash
docker build -t hateshield .
docker run -p 5000:5000 hateshield
```

---

## Next Steps

1. **Choose deployment option** (Local/Gunicorn/Docker)
2. **Update config.py** for production settings
3. **Test all 3 models** (LR, LSTM, MuRIL)
4. **Set up monitoring** (logging, alerts)
5. **Deploy to server**

---

**Ready to proceed with which option?**

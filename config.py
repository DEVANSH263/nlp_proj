import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # PostgreSQL on Railway (auto-set by Railway)
    # Falls back to SQLite locally
    database_url = os.environ.get('DATABASE_URL')
    if database_url:
        SQLALCHEMY_DATABASE_URI = database_url.replace('postgresql://', 'postgresql+psycopg2://')
    else:
        SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'database.db')
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # All models stored locally
    MODEL_PATH = os.path.join(BASE_DIR, 'model3', 'lr', 'model.pkl')
    VECTORIZER_PATH = os.path.join(BASE_DIR, 'model3', 'lr', 'vectorizer.pkl')
    LSTM_MODEL_PATH = os.path.join(BASE_DIR, 'model3', 'lstm', 'lstm_model.pt')
    LSTM_VOCAB_PATH = os.path.join(BASE_DIR, 'model3', 'lstm', 'lstm_vocab.pkl')
    MURIL_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'muril')

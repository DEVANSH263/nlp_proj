from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User account model."""
    __tablename__ = 'users'

    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    # Relationship: one user → many predictions
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'


class Prediction(db.Model):
    """Stores every prediction made by a logged-in user."""
    __tablename__ = 'predictions'

    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    input_text      = db.Column(db.Text, nullable=False)
    normalized_text = db.Column(db.Text, nullable=True)   # None when normalization off
    prediction      = db.Column(db.String(10), nullable=False)  # 'HOF' or 'NOT'
    confidence      = db.Column(db.Float, nullable=False)
    timestamp       = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Prediction {self.id} [{self.prediction}]>'

import io
import csv

from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from datetime import datetime

from models import db, Prediction
from utils.predict import predict

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Root URL – redirect to home."""
    return redirect(url_for('main.home'))


@main_bp.route('/home')
def home():
    """Public landing page."""
    return render_template('home.html')


@main_bp.route('/about')
def about():
    """About / info page."""
    return render_template('about.html')


@main_bp.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    """Main dashboard: text input → prediction display."""
    result = None  # holds prediction dict when POST succeeds

    if request.method == 'POST':
        input_text = request.form.get('input_text', '').strip()

        if not input_text:
            flash('Please enter some text to analyse.', 'warning')
            return render_template('dashboard.html', result=None)

        if len(input_text) > 2000:
            flash('Input too long. Please limit to 2000 characters.', 'warning')
            return render_template('dashboard.html', result=None)

        model_type = request.form.get('model_type', 'lr')   # 'lr', 'lstm', or 'muril'

        # Run prediction pipeline (model-specific preprocessing is built-in)
        result = predict(input_text, model_type=model_type)

        # Persist to database
        pred_record = Prediction(
            user_id    = current_user.id,
            input_text = input_text,
            prediction = result['prediction'],
            confidence = result['confidence'],
        )
        db.session.add(pred_record)
        db.session.commit()
        # Pass the saved ID so the template can link to the single-prediction report
        result['pred_id'] = pred_record.id

    return render_template('dashboard.html', result=result)


@main_bp.route('/history')
@login_required
def history():
    """Show all past predictions for the current user, newest first."""
    predictions = (
        Prediction.query
        .filter_by(user_id=current_user.id)
        .order_by(Prediction.timestamp.desc())
        .all()
    )
    return render_template('history.html', predictions=predictions)


@main_bp.route('/compare', methods=['GET', 'POST'])
@login_required
def compare():
    """Run all 3 models on the same text and show side-by-side results."""
    results = None
    input_text = ''
    if request.method == 'POST':
        input_text = request.form.get('input_text', '').strip()
        if not input_text:
            flash('Please enter some text to compare.', 'warning')
        elif len(input_text) > 2000:
            flash('Input too long. Please limit to 2000 characters.', 'warning')
        else:
            # Run all 3 models with their model-specific preprocessing
            results = {mt: predict(input_text, model_type=mt)
                       for mt in ['lr', 'lstm', 'muril']}
    return render_template('compare.html', results=results, input_text=input_text)


@main_bp.route('/batch', methods=['GET', 'POST'])
@login_required
def batch():
    """Batch-analyse multiple texts from a textarea or uploaded CSV."""
    rows = None
    model_type = request.form.get('model_type', 'lr')

    if request.method == 'POST':
        texts = []

        # --- CSV file upload ---
        f = request.files.get('csv_file')
        if f and f.filename:
            raw_bytes = f.read()
            content = raw_bytes.decode('utf-8-sig')
            reader = csv.reader(io.StringIO(content))
            header = next(reader, None)
            text_col = 0
            if header:
                for i, h in enumerate(header):
                    if h.strip().lower() in ('text', 'content', 'message'):
                        text_col = i
                        break
                else:
                    # No recognised header → first row is data
                    if header[text_col].strip():
                        texts.append(header[text_col].strip())
            for row in reader:
                if row and len(row) > text_col and row[text_col].strip():
                    texts.append(row[text_col].strip())
        else:
            # --- Textarea fallback (one line per text) ---
            raw = request.form.get('paste_text', '').strip()
            texts = [l.strip() for l in raw.splitlines() if l.strip()]

        texts = texts[:200]  # hard cap

        if not texts:
            flash('No texts found. Paste lines or upload a valid CSV.', 'warning')
            return render_template('batch.html', rows=None, model_type=model_type)

        rows = []
        for t in texts:
            r = predict(t, model_type=model_type)
            rows.append({
                'text':       t,
                'prediction': r['prediction'],
                'confidence': r['confidence'],
                'model_used': r['model_used'],
            })

    return render_template('batch.html', rows=rows, model_type=model_type)

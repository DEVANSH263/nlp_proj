import sys
sys.path.insert(0, r'c:\Documents\Projects\ss')
from app import create_app
app = create_app()
with app.app_context():
    from utils.predict import predict
    text = 'yaar tu sach mein bohot achha hai, keep it up!'
    for m in ['lr', 'lstm', 'muril']:
        r = predict(text, model_type=m)
        print(m.upper(), '->', r['prediction'], 'conf=', r['confidence'])

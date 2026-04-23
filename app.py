import logging
from flask import Flask
from flask_login import LoginManager
from config import Config
from models import db, User

logging.basicConfig(level=logging.DEBUG)

login_manager = LoginManager()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialise extensions
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'warning'

    # User loader for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register blueprints
    from routes.auth   import auth_bp
    from routes.main   import main_bp
    from routes.report import report_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(report_bp)

    # Create all DB tables on first run
    with app.app_context():
        db.create_all()

    return app


if __name__ == '__main__':
    application = create_app()
    application.run(
        debug=True,
        use_reloader=False,
    )

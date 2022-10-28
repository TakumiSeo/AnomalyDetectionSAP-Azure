from flask import Flask
from flask_cors import CORS
import os

def create_app():
    # Generate Flask App Instance
    app = Flask(__name__)
    app.config['MAX_COOKIE_SIZE'] = 0
    # Register Router instance
    with app.app_context():
        import router
        app.register_blueprint(router.router)
    app.config.from_mapping(
        SECRET_KEY='temp',
    )
    
    CORS(
        app,
        support_credentials=True,
    )

    return app

app = create_app()
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8001, use_reloader=True)
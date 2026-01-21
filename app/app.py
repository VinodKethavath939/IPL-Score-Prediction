"""
Flask Application Entrypoint
============================
This file serves as the entrypoint for Flask deployment.
"""

from flask_app import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

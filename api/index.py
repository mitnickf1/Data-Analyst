"""
api/index.py
Entry point untuk Vercel Serverless Function.
Vercel secara otomatis mendeteksi handler Flask dari variabel `app`.
"""

import sys
import os

# Tambahkan direktori root proyek ke sys.path agar app.py dan modul pendukungnya dapat diimpor
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app

class VercelPathFixMiddleware:
    """
    Middleware WSGI untuk menyesuaikan PATH_INFO di lingkungan Vercel Serverless.
    Vercel sering meneruskan PATH_INFO sebagai '/api/index.py' atau '/api',
    sementara path sebenarnya tersimpan di HTTP_X_MATCHED_PATH.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        matched_path = environ.get("HTTP_X_MATCHED_PATH")
        if matched_path:
            environ["PATH_INFO"] = matched_path
        else:
            path_info = environ.get("PATH_INFO", "")
            for prefix in ["/api/index.py", "/api/index", "/api"]:
                if path_info == prefix:
                    environ["PATH_INFO"] = "/"
                    break
                elif path_info.startswith(prefix + "/"):
                    environ["PATH_INFO"] = path_info[len(prefix):]
                    break
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)

# Handler WSGI untuk serverless Vercel
handler = app


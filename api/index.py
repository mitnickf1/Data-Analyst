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
    Middleware WSGI untuk membersihkan SCRIPT_NAME dan PATH_INFO di Vercel.
    Memastikan url_for() selalu menghasilkan URL bersih (/analyze bukan /api/index.py/analyze)
    dan semua variasi prefix dinormalkan ke rute asli Flask.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        environ["SCRIPT_NAME"] = ""
        path_info = environ.get("PATH_INFO", "")
        while True:
            changed = False
            for prefix in ["/api/index.py", "/api/index", "/api"]:
                if path_info == prefix or path_info == prefix + "/":
                    path_info = "/"
                    changed = True
                    break
                elif path_info.startswith(prefix + "/"):
                    path_info = path_info[len(prefix):]
                    changed = True
                    break
            if not changed:
                break
        environ["PATH_INFO"] = path_info or "/"
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)

# Handler WSGI untuk serverless Vercel
handler = app


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

# Handler WSGI untuk serverless Vercel
handler = app

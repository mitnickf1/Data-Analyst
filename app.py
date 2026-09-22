"""
app.py — Website sederhana untuk Autonomous EDA Agent.

Jalankan dengan:
    python app.py
lalu buka http://127.0.0.1:5000 di browser.
"""

import os
import uuid
import math
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session, send_from_directory

from eda_agent import EDAAgent
from llm_narrator import (
    generate_llm_narrative,
    parse_prompt_locally,
    parse_prompt_to_spec,
    LLMNarratorError,
)

import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Gunakan tempdir OS/serverless (/tmp di Linux/Vercel) agar terhindar dari read-only filesystem
UPLOAD_DIR = os.environ.get("UPLOAD_DIR") or os.path.join(tempfile.gettempdir(), "eda_agent_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {"csv", "xlsx", "xls"}
MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = os.environ.get("SECRET_KEY", "eda-agent-secret-key")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# cache sederhana in-memory untuk hasil analisis terakhir (agar bisa didownload)
LAST_RESULT_HTML = {"html": None}


@app.template_filter("fmt4")
def fmt4(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(value):
        return "-"
    return f"{value:.4f}"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _read_css():
    css_path = os.path.join(BASE_DIR, "static", "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        return f.read()


@app.route("/static/<path:filename>")
@app.route("/api/static/<path:filename>")
@app.route("/api/index/static/<path:filename>")
@app.route("/api/index.py/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)



def _run_llm_narrative_layer(results, form):
    """Jalankan lapisan narasi LLM opsional. Selalu fallback aman jika gagal."""
    api_key = (form.get("gemini_api_key", "") or os.environ.get("GEMINI_API_KEY", "")).strip("'\" \t\r\n")
    if not api_key:
        return None, None
    try:
        narrative = generate_llm_narrative(
            results["profile"], results["patterns"], results["tests"], api_key
        )
        return narrative, None
    except LLMNarratorError as e:
        return None, str(e)


def _render_full_report(results, filename, llm_narrative, llm_error, custom_result=None,
                         custom_prompt_used=None):
    common = dict(
        filename=filename,
        profile=results["profile"],
        patterns=results["patterns"],
        tests=results["tests"],
        charts=results["charts"],
        narrative=results["narrative"],
        llm_narrative=llm_narrative,
        llm_error=llm_error,
        custom_result=custom_result,
        custom_prompt_used=custom_prompt_used,
    )
    rendered = render_template("report.html", **common)
    LAST_RESULT_HTML["html"] = render_template("report_standalone.html", css=_read_css(), **common)
    return rendered


@app.route("/", methods=["GET"])
@app.route("/api", methods=["GET"])
@app.route("/api/index", methods=["GET"])
@app.route("/api/index.py", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
@app.route("/api/analyze", methods=["POST"])
@app.route("/api/index/analyze", methods=["POST"])
@app.route("/api/index.py/analyze", methods=["POST"])
def analyze():
    file = request.files.get("dataset")
    if not file or file.filename == "":
        flash("Silakan pilih file CSV atau Excel terlebih dahulu.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Format file tidak didukung. Gunakan .csv, .xlsx, atau .xls")
        return redirect(url_for("index"))

    # Simpan file secara PERSISTEN (tidak langsung dihapus) supaya bisa dipakai
    # ulang saat user menjalankan uji manual/prompt tambahan di halaman laporan.
    ext = file.filename.rsplit(".", 1)[1].lower()
    persistent_name = f"{uuid.uuid4().hex}.{ext}"
    persistent_path = os.path.join(UPLOAD_DIR, persistent_name)
    file.save(persistent_path)

    try:
        agent = EDAAgent(persistent_path)
        results = agent.run_full_analysis()
    except Exception as e:
        flash(f"Gagal menganalisis file: {e}")
        if os.path.exists(persistent_path):
            os.remove(persistent_path)
        return redirect(url_for("index"))

    # simpan referensi dataset di session (cookie) supaya request berikutnya
    # (uji manual/prompt) tahu file mana yang harus dibaca ulang
    session["dataset_path"] = persistent_path
    session["dataset_filename"] = file.filename

    llm_narrative, llm_error = _run_llm_narrative_layer(results, request.form)
    return _render_full_report(results, file.filename, llm_narrative, llm_error)


@app.route("/custom_test", methods=["POST"])
@app.route("/api/custom_test", methods=["POST"])
@app.route("/api/index/custom_test", methods=["POST"])
@app.route("/api/index.py/custom_test", methods=["POST"])
def custom_test():
    """
    Jalankan SATU uji statistik tambahan di atas dataset yang sudah diupload,
    dipicu secara MANUAL (dropdown) atau lewat MODE PROMPT (bahasa natural,
    diterjemahkan oleh Gemini menjadi spesifikasi uji).
    """
    dataset_path = session.get("dataset_path")
    filename = session.get("dataset_filename", "dataset")

    if not dataset_path or not os.path.exists(dataset_path):
        flash("Sesi dataset sudah tidak ada. Silakan upload ulang file Anda.")
        return redirect(url_for("index"))

    try:
        agent = EDAAgent(dataset_path)
        results = agent.run_full_analysis()
    except Exception as e:
        flash(f"Gagal memuat ulang dataset: {e}")
        return redirect(url_for("index"))

    prompt_text = request.form.get("prompt_text", "").strip()
    custom_prompt_used = None
    custom_result = None

    if prompt_text:
        # --- MODE PROMPT: minta Gemini menerjemahkan instruksi jadi spesifikasi uji ---
        custom_prompt_used = prompt_text
        api_key = (request.form.get("gemini_api_key_custom", "") or os.environ.get("GEMINI_API_KEY", "")).strip("'\" \t\r\n")
        column_info = {
            "numeric_cols": results["profile"]["numeric_cols"],
            "categorical_cols": results["profile"]["categorical_cols"],
        }
        if not api_key:
            spec = parse_prompt_locally(prompt_text, column_info)
            if spec.get("error"):
                custom_result = {"error": f"Pertanyaan belum dikenali: {spec['error']}"}
            else:
                custom_result = agent.run_custom_test(spec)
                if "error" not in custom_result:
                    custom_result["assumption_note"] = (
                        (custom_result.get("assumption_note") or "")
                        + " Analisis dijalankan secara lokal oleh Autonomous EDA Agent."
                    ).strip()
        else:
            try:
                spec = parse_prompt_to_spec(prompt_text, column_info, api_key)
                if spec.get("error"):
                    custom_result = {"error": spec["error"]}
                else:
                    custom_result = agent.run_custom_test(spec)
            except LLMNarratorError as e:
                spec = parse_prompt_locally(prompt_text, column_info)
                if spec.get("error"):
                    custom_result = {"error": f"Gemini tidak merespons ({e}). Analisis lokal juga belum mengenali pertanyaan: {spec['error']}"}
                else:
                    custom_result = agent.run_custom_test(spec)
                    if "error" not in custom_result:
                        custom_result["assumption_note"] = (
                            (custom_result.get("assumption_note") or "")
                            + f" (Dialihkan ke analisis lokal Autonomous EDA Agent karena kendala Gemini API: {e})."
                        ).strip()
    else:
        # --- MODE MANUAL: user memilih langsung lewat dropdown ---
        spec = {
            "test": request.form.get("test_type", "auto"),
            "col1": request.form.get("col1") or None,
            "col2": request.form.get("col2") or None,
        }
        custom_result = agent.run_custom_test(spec)

    llm_narrative, llm_error = _run_llm_narrative_layer(results, request.form)
    return _render_full_report(results, filename, llm_narrative, llm_error,
                                custom_result=custom_result, custom_prompt_used=custom_prompt_used)


@app.route("/download_report")
@app.route("/api/download_report")
@app.route("/api/index/download_report")
@app.route("/api/index.py/download_report")
def download_report():
    if not LAST_RESULT_HTML["html"]:
        flash("Belum ada laporan yang bisa diunduh. Silakan jalankan analisis dulu.")
        return redirect(url_for("index"))
    path = os.path.join(UPLOAD_DIR, "laporan_eda.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(LAST_RESULT_HTML["html"])
    return send_file(path, as_attachment=True, download_name="laporan_eda.html")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

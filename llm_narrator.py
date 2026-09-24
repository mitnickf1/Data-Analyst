"""
llm_narrator.py
================
Lapisan LLM opsional untuk Autonomous EDA Agent, memakai Google Gemini API.

Prinsip desain:
- Bersifat OPSIONAL. Jika API key tidak diisi atau panggilan API gagal
  (timeout, kuota habis, key salah, dsb), sistem tetap jalan memakai
  narasi rule-based dari eda_agent.py (tidak pernah "mati total").
- Agent mengirim RINGKASAN hasil analisis (bukan seluruh dataset mentah)
  ke Gemini, supaya: (1) hemat token, (2) data mentah pengguna tidak perlu
  dikirim ke luar jika sensitif — hanya statistik agregatnya.

Endpoint: Gemini Interactions API (generativelanguage.googleapis.com), model default
'gemini-3.7-flash'.
"""

import os
import json
import re
import difflib
import requests

GEMINI_MODEL_DEFAULT = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MODEL_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]
GEMINI_INTERACTIONS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
REQUEST_TIMEOUT = 15  # detik per percobaan


class LLMNarratorError(Exception):
    """Dilempar saat panggilan ke Gemini API gagal (network, auth, kuota, dll)."""


def _build_summary_payload(profile: dict, patterns: dict, tests: list) -> dict:
    """
    Ringkas hasil analisis menjadi struktur kecil sebelum dikirim ke LLM.
    Hanya statistik agregat yang dikirim — bukan baris data mentah.
    """
    top_tests = sorted(tests, key=lambda t: t["p_value"])[:12]
    return {
        "jumlah_baris": profile["n_rows"],
        "jumlah_kolom": profile["n_cols"],
        "kolom_numerik": profile["numeric_cols"],
        "kolom_kategorikal": profile["categorical_cols"],
        "kolom_duplikat": profile["duplicate_rows"],
        "kolom_missing": profile["missing"],
        "ringkasan_numerik": profile["numeric_summary"],
        "outlier": patterns["outliers"],
        "korelasi_kuat": patterns["strong_correlations"],
        "kolom_skewed": patterns["skewed_cols"],
        "kolom_konstan": patterns["constant_cols"],
        "hasil_uji_statistik": [
            {
                "metode": t["method"],
                "variabel": t["variables"],
                "p_value": round(t["p_value"], 5),
                "signifikan": t["significant"],
            }
            for t in top_tests
        ],
    }


def _build_prompt(summary: dict) -> str:
    return f"""Kamu adalah seorang data analyst senior yang membuat ringkasan eksekutif dari hasil
exploratory data analysis (EDA) otomatis. Berikut hasil analisis statistik dalam format JSON:

{json.dumps(summary, ensure_ascii=False, indent=2)}

Tulis ringkasan analisis dalam Bahasa Indonesia yang:
1. Mudah dipahami orang non-statistik (hindari jargon berlebihan, jelaskan istilah bila perlu)
2. Fokus pada insight yang PALING PENTING dan actionable, bukan mengulang semua angka
3. Menyoroti risiko/perhatian pada kualitas data (missing value, outlier) bila relevan
4. Menjelaskan makna praktis dari hasil uji statistik yang signifikan (bukan hanya menyebut p-value)
5. Ditutup dengan 2-3 rekomendasi konkret langkah selanjutnya

Format keluaran: paragraf naratif mengalir (boleh dengan sub-heading singkat), TANPA
menyebutkan bahwa kamu adalah AI atau menyebut proses pembuatan ringkasan ini.
Panjang maksimal sekitar 300-400 kata."""


def _extract_interaction_text(data: dict) -> str:
    """Ambil teks dari respons Google Gemini API."""
    if not isinstance(data, dict):
        return ""

    for key in ("output_text", "outputText"):
        text = data.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()

    interaction = data.get("interaction")
    if isinstance(interaction, dict):
        for key in ("output_text", "outputText"):
            text = interaction.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()

    candidates = data.get("candidates", [])
    if candidates and isinstance(candidates, list):
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        if text:
            return text

    steps = data.get("steps", [])
    if isinstance(steps, list):
        for step in reversed(steps):
            if isinstance(step, dict) and step.get("type") == "model_output":
                content = step.get("content", [])
                if isinstance(content, list):
                    text = "".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ).strip()
                    if text:
                        return text

    return ""


def _extract_json_object(text: str) -> dict:
    """Parse JSON langsung, atau ekstrak objek JSON dari output model."""
    text = str(text).strip()
    # Bersihkan markdown code block jika ada
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if m:
            text = m.group(1).strip()
        else:
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Coba ambil karakter pertama '{' hingga '}' terluar
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
        raise LLMNarratorError(f"Format keluaran model bukan JSON valid: {text[:150]}")


def _extract_google_error(resp: requests.Response) -> str:
    """Ambil pesan error eksplisit yang dikembalikan oleh Google Gemini API."""
    try:
        data = resp.json()
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict):
            err = data.get("error", {})
            msg = err.get("message")
            status = err.get("status", "")
            if msg:
                return f"{msg} (Status: {status})" if status else msg
    except Exception:
        pass
    return resp.text[:250] if resp.text else f"Status HTTP {resp.status_code}"


def _post_gemini(model: str, prompt: str, api_key: str, *, json_output=False) -> str:
    """
    Panggil Gemini API dengan endpoint universal generateContent.
    Mengembalikan teks respons atau melempar LLMNarratorError dengan pesan jelas.
    """
    api_key = str(api_key or "").strip("'\" \t\r\n")
    if not api_key:
        raise LLMNarratorError("API key Gemini tidak boleh kosong.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    generation_config = {
        "maxOutputTokens": 1024 if not json_output else 512,
        "temperature": 0.2,
    }

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            txt = _extract_interaction_text(resp.json())
            if txt:
                return txt
            raise LLMNarratorError(f"Gemini ({model}) berhasil dihubungi namun tidak mengembalikan teks.")
        else:
            err_detail = _extract_google_error(resp)
            if resp.status_code in (400, 401, 403) and any(
                kw in err_detail.lower() for kw in ["api key", "api_key", "identity", "unregistered", "invalid_argument"]
            ):
                raise LLMNarratorError(f"API key Gemini tidak valid: {err_detail}")
            if resp.status_code == 429:
                raise LLMNarratorError(f"Batas kuota Gemini API terlampaui: {err_detail}")
            raise LLMNarratorError(f"Gemini API ({model}) HTTP {resp.status_code}: {err_detail}")
    except requests.exceptions.RequestException as e:
        raise LLMNarratorError(f"Gagal menghubungi Gemini API ({model}): {e}")


def generate_llm_narrative(profile: dict, patterns: dict, tests: list,
                            api_key: str, model: str = None) -> str:
    """
    Kirim ringkasan hasil EDA ke Gemini API dan kembalikan narasi teks.
    Mencoba model default, lalu fallback models jika terjadi timeout / kendala teknis.
    """
    if not api_key:
        raise LLMNarratorError("API key Gemini tidak diisi.")

    candidate_models = [model] if model else [GEMINI_MODEL_DEFAULT] + GEMINI_MODEL_FALLBACKS
    summary = _build_summary_payload(profile, patterns, tests)
    prompt = _build_prompt(summary)
    last_error = None

    for candidate in candidate_models:
        try:
            return _post_gemini(candidate, prompt, api_key, json_output=False)
        except LLMNarratorError as e:
            last_error = e
            continue

    raise last_error or LLMNarratorError("Gagal memanggil Gemini API.")


# ---------------------------------------------------------------------- #
# MODE PROMPT: terjemahkan instruksi bahasa natural -> spesifikasi uji
# ---------------------------------------------------------------------- #

VALID_TESTS = [
    "ttest", "welch", "mannwhitney", "anova", "kruskal",
    "chi2", "pearson", "spearman", "auto",
    "column_stats", "distribution", "normality", "category_stats",
    "numeric_compare", "dataset_overview", "missing_values",
    "outlier_report", "top_correlations", "top_finding", "dataset_summary",
    "direct_answer",
]


def match_columns_in_prompt(prompt: str, numeric_cols: list, categorical_cols: list):
    """
    Deteksi nama kolom dalam prompt pengguna secara fleksibel:
    - Case-insensitive
    - Penanganan spasi dan underscore (e.g. 'detergents paper' -> 'Detergents_Paper')
    - Sub-kata dari kolom majemuk (e.g. 'detergent' -> 'Detergents_Paper')
    - Fuzzy matching dengan toleransi typo ringan menggunakan difflib
    """
    prompt_clean = prompt.lower()
    all_cols = list(numeric_cols) + list(categorical_cols)
    found = []

    # 1. Exact match / substring / compound word parts
    for col in sorted(all_cols, key=len, reverse=True):
        col_clean = str(col).lower().replace("_", " ")
        raw_col = str(col).lower()
        if re.search(r"\b" + re.escape(raw_col) + r"\b", prompt_clean) or re.search(r"\b" + re.escape(col_clean) + r"\b", prompt_clean):
            if col not in found:
                found.append(col)
        elif raw_col in prompt_clean or col_clean in prompt_clean:
            if col not in found:
                found.append(col)

        # Cek potongan kata dari nama kolom majemuk
        parts = re.split(r"[_ ]+", str(col).lower())
        for p in parts:
            if len(p) >= 4 and p in prompt_clean:
                if col not in found:
                    found.append(col)

    # 2. Fuzzy matching kata per kata
    tokens = re.findall(r"[a-zA-Z_]+", prompt_clean)
    all_targets = {}
    for c in all_cols:
        all_targets[str(c).lower()] = c
        for p in re.split(r"[_ ]+", str(c).lower()):
            if len(p) >= 4:
                all_targets[p] = c

    for t in tokens:
        if len(t) >= 4 and t not in [str(c).lower() for c in found]:
            matches = difflib.get_close_matches(t, all_targets.keys(), n=1, cutoff=0.75)
            if matches:
                matched_col = all_targets[matches[0]]
                if matched_col not in found:
                    found.append(matched_col)

    # Urutkan berdasarkan posisi kemunculan pertama di prompt
    def pos(c):
        p1 = prompt_clean.find(str(c).lower())
        p2 = prompt_clean.find(str(c).lower().replace("_", " "))
        candidates = [p for p in (p1, p2) if p != -1]
        return min(candidates) if candidates else 999

    found.sort(key=pos)
    num_found = [c for c in found if c in numeric_cols]
    cat_found = [c for c in found if c in categorical_cols]
    return num_found, cat_found


def parse_prompt_locally(user_prompt: str, column_info: dict) -> dict:
    """
    Fallback lokal cerdas jika Gemini API tidak tersedia atau timeout.
    Mampu menjawab aneka pertanyaan analisis data:
    - Statistik deskriptif (rata-rata, median, max, min, sum, std)
    - Bentuk distribusi & uji normalitas
    - Analisis kategori & frekuensi
    - Perbandingan 2 kolom numerik
    - Uji korelasi
    - Uji komparasi grup (kategorikal vs numerik)
    - Uji asosiasi (Chi-square)
    - Informasi kualitas data (missing values, outlier)
    - Dimensi dataset (jumlah baris & kolom)
    - Korelasi terkuat
    - Ringkasan dataset & kesimpulan
    """
    prompt = user_prompt.lower().strip()
    numeric_cols = column_info.get("numeric_cols", [])
    categorical_cols = column_info.get("categorical_cols", [])

    mentioned_numeric, mentioned_categorical = match_columns_in_prompt(prompt, numeric_cols, categorical_cols)

    # 0. Pertanyaan "1 data penting" / "temuan penting" / "rekomendasi"
    if any(w in prompt for w in [
        "1 data penting", "satu data penting", "data penting", "temuan penting",
        "hal terpenting", "faktor penting", "yang paling penting", "insight penting",
        "rekomendasi utama", "temuan utama", "insight utama"
    ]):
        return {"test": "top_finding", "col1": None, "col2": None, "error": None}

    # 1. Dataset overview / Dimensi data
    if any(w in prompt for w in [
        "berapa baris", "jumlah baris", "banyak baris", "total baris",
        "jumlah kolom", "berapa kolom", "dimensi", "ukuran data",
        "ukuran dataset", "banyak data", "jumlah data", "berapa data"
    ]) and not mentioned_numeric and not mentioned_categorical:
        return {"test": "dataset_overview", "col1": None, "col2": None, "error": None}

    # 2. Missing values / Kualitas data
    if any(w in prompt for w in ["missing", "kosong", "null", "hilang", "kelengkapan data", "data kosong"]):
        target = mentioned_numeric[0] if mentioned_numeric else (mentioned_categorical[0] if mentioned_categorical else None)
        return {"test": "missing_values", "col1": target, "col2": None, "error": None}

    # 3. Outliers / Pencilan
    if any(w in prompt for w in ["outlier", "pencilan", "anomali", "ekstrem"]):
        target = mentioned_numeric[0] if mentioned_numeric else None
        return {"test": "outlier_report", "col1": target, "col2": None, "error": None}

    # 4. Top correlations
    if any(w in prompt for w in [
        "korelasi tertinggi", "korelasi terkuat", "hubungan terkuat",
        "korelasi terbesar", "top korelasi", "korelasi apa saja",
        "korelasi paling", "hubungan paling kuat"
    ]):
        return {"test": "top_correlations", "col1": None, "col2": None, "error": None}

    # 5. Dataset Summary / Insight Umum
    if any(w in prompt for w in [
        "ringkasan", "kesimpulan", "rangkuman", "insight",
        "jelaskan dataset", "tentang data", "apa isi data",
        "temuan utama", "overview", "ikhtisar", "makna data"
    ]) and not mentioned_numeric:
        return {"test": "dataset_summary", "col1": None, "col2": None, "error": None}

    # 6. Uji Normalitas & Distribusi
    if any(w in prompt for w in ["normal", "normalitas", "shapiro", "kemiringan", "skew"]):
        target = mentioned_numeric[0] if mentioned_numeric else (numeric_cols[0] if numeric_cols else None)
        return {"test": "normality", "col1": target, "col2": None, "error": None}

    if any(w in prompt for w in ["distribusi", "sebaran", "histogram", "persebaran", "densitas"]):
        if mentioned_numeric:
            return {"test": "distribution", "col1": mentioned_numeric[0], "col2": None, "error": None}
        if mentioned_categorical:
            return {"test": "category_stats", "col1": mentioned_categorical[0], "col2": None, "error": None}
        return {"test": "distribution", "col1": numeric_cols[0] if numeric_cols else None, "col2": None, "error": None}

    # 7. Statistik Deskriptif (Rata-rata, Median, Min, Max, Sum, dsb.)
    stat_keywords = {
        "mean": ["rata-rata", "rata rata", "mean", "average", "rerata"],
        "median": ["median", "nilai tengah"],
        "max": ["tertinggi", "maksimum", "maksimal", "terbesar", "paling tinggi", "max", "puncak"],
        "min": ["terendah", "minimum", "minimal", "terkecil", "paling rendah", "min", "dasar"],
        "sum": ["total", "jumlah total", "sum", "penjumlahan", "akumulasi"],
        "std": ["standar deviasi", "deviasi", "std", "simpangan baku", "variasi"],
    }
    found_stat = None
    for stat_name, kws in stat_keywords.items():
        if any(kw in prompt for kw in kws):
            found_stat = stat_name
            break

    if found_stat or any(w in prompt for w in ["kuartil", "iqr", "varians", "deskriptif", "statistik"]):
        stat_focus = found_stat or "all"
        if mentioned_numeric:
            return {"test": "column_stats", "col1": mentioned_numeric[0], "col2": stat_focus, "error": None}
        if mentioned_categorical:
            return {"test": "category_stats", "col1": mentioned_categorical[0], "col2": None, "error": None}
        if any(w in prompt for w in ["semua", "seluruh", "setiap"]):
            return {"test": "column_stats", "col1": "all", "col2": stat_focus, "error": None}

    # 8. Analisis Kategori & Frekuensi
    if any(w in prompt for w in ["kategori", "frekuensi", "proporsi", "paling banyak", "terbanyak", "dominan", "persebaran kategori"]):
        if mentioned_categorical:
            return {"test": "category_stats", "col1": mentioned_categorical[0], "col2": None, "error": None}

    # 9. Korelasi (2 kolom numerik)
    if any(w in prompt for w in ["korelasi", "correlation", "hubungan", "relasi"]):
        test = "spearman" if any(w in prompt for w in ["spearman", "rank", "non-parametrik", "nonparametrik"]) else "pearson"
        if len(mentioned_numeric) >= 2:
            return {"test": test, "col1": mentioned_numeric[0], "col2": mentioned_numeric[1], "error": None}
        if len(mentioned_numeric) == 1:
            other = [c for c in numeric_cols if c != mentioned_numeric[0]]
            if other:
                return {"test": test, "col1": mentioned_numeric[0], "col2": other[0], "error": None}
        return {"test": "top_correlations", "col1": None, "col2": None, "error": None}

    # 10. Asosiasi (2 kolom kategorikal)
    if any(w in prompt for w in ["chi", "asosiasi", "association"]) or (
        len(mentioned_categorical) >= 2 and any(w in prompt for w in ["bandingkan", "beda", "hubungan", "kaitkan"])
    ):
        if len(mentioned_categorical) >= 2:
            return {"test": "chi2", "col1": mentioned_categorical[0], "col2": mentioned_categorical[1], "error": None}

    # 11. Perbandingan 2 kolom numerik
    if len(mentioned_numeric) >= 2 and any(w in prompt for w in ["bandingkan", "beda", "perbedaan", "versus", "vs", "lebih besar", "komparasi"]):
        return {"test": "numeric_compare", "col1": mentioned_numeric[0], "col2": mentioned_numeric[1], "error": None}

    # 12. Uji Komparasi Grup (1 kategorikal, 1 numerik — urutan bebas)
    if mentioned_categorical and mentioned_numeric:
        cat_col = mentioned_categorical[0]
        num_col = mentioned_numeric[0]
        if "mann" in prompt:
            test = "mannwhitney"
        elif "welch" in prompt:
            test = "welch"
        elif "anova" in prompt:
            test = "anova"
        elif "kruskal" in prompt:
            test = "kruskal"
        elif "t-test" in prompt or "ttest" in prompt:
            test = "ttest"
        else:
            test = "auto"
        return {"test": test, "col1": cat_col, "col2": num_col, "error": None}

    # 13. Default jika 1 kolom disebut
    if mentioned_numeric:
        return {"test": "column_stats", "col1": mentioned_numeric[0], "col2": "all", "error": None}
    if mentioned_categorical:
        return {"test": "category_stats", "col1": mentioned_categorical[0], "col2": None, "error": None}

    # 14. Fallback umum: sajikan ringkasan dataset
    return {"test": "dataset_summary", "col1": None, "col2": None, "error": None}


def _build_prompt_spec_instruction(user_prompt: str, column_info: dict) -> str:
    return f"""Kamu adalah asisten analis data cerdas. Pengguna menulis pertanyaan atau instruksi analisis:
"{user_prompt}"

Kolom yang tersedia di dataset:
- Numerik: {column_info.get('numeric_cols', [])}
- Kategorikal: {column_info.get('categorical_cols', [])}

PILIH SALAH SATU FORMAT RESPON JSON BERIKUT:

1. Jika pengguna meminta uji statistik, grafik, atau komparasi tertentu (misal: "korelasi Fresh dan Milk", "rata-rata Milk", "distribusi Channel", "outlier"):
   {{"test": "column_stats"|"distribution"|"normality"|"category_stats"|"numeric_compare"|"ttest"|"anova"|"chi2"|"pearson"|"outlier_report"|"missing_values"|"top_correlations"|"top_finding"|"dataset_summary", "col1": "nama_kolom", "col2": "nama_kolom_atau_null", "error": null}}

2. Jika pengguna meminta penjelasan naratif, insight khusus, interpretasi bisnis, atau pertanyaan terbuka (contoh: "jelaskan 1 data penting", "mengapa data demikian", "apa strategi terbaik"):
   {{"test": "direct_answer", "method": "Jawaban Analisis AI Cerdas", "variables": "{user_prompt}", "answer": "<Tuliskan penjelasan analitis yang tajam, mendalam, dan langsung menjawab pertanyaan pengguna secara to-the-point>", "metrics": [{{"label": "Fokus", "value": "Temuan Kritis"}}], "error": null}}

Jawab HANYA dengan format JSON valid persis."""


def parse_prompt_to_spec(user_prompt: str, column_info: dict, api_key: str,
                          model: str = None) -> dict:
    """
    Kirim instruksi bahasa natural pengguna ke Gemini, minta dikembalikan
    sebagai JSON spesifikasi analisis/uji statistik: {test, col1, col2, error}.
    """
    if not api_key:
        raise LLMNarratorError("API key Gemini tidak diisi.")

    candidate_models = [model] if model else [GEMINI_MODEL_DEFAULT] + GEMINI_MODEL_FALLBACKS
    instruction = _build_prompt_spec_instruction(user_prompt, column_info)
    last_error = None

    for candidate in candidate_models:
        try:
            text = _post_gemini(candidate, instruction, api_key, json_output=True)
            spec = _extract_json_object(text)
            if spec.get("test") not in VALID_TESTS:
                last_error = LLMNarratorError(f"Gemini mengembalikan jenis uji tidak dikenal: {spec.get('test')}")
                continue
            return spec
        except (LLMNarratorError, KeyError, IndexError, json.JSONDecodeError, TypeError) as e:
            last_error = LLMNarratorError(f"Gagal mem-parsing respons Gemini: {e}")
            continue

    raise last_error or LLMNarratorError("Gagal memanggil Gemini API.")


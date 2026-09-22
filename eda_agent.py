"""
eda_agent.py
=============
Autonomous Exploratory Data Analyst (EDA) Agent.

Menerima CSV/Excel mentah, lalu secara otomatis:
1. Memprofilkan data (tipe kolom, missing value, distribusi)
2. Mendeteksi pola (outlier, korelasi, skewness)
3. Memilih & menjalankan uji statistik yang sesuai (t-test, Welch, Mann-Whitney,
   ANOVA, Kruskal-Wallis, Chi-square, Pearson) berdasarkan karakteristik data
   (jumlah grup, normalitas, homogenitas varians)
4. Menghasilkan visualisasi (base64 PNG) dan laporan naratif otomatis

Desain: setiap keputusan statistik didasarkan pada uji asumsi terlebih dahulu
(Shapiro-Wilk untuk normalitas, Levene untuk homogenitas varians), sehingga
agent ini meniru alur berpikir seorang analis data manusia.
"""

import io
import base64
import math
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", font_scale=0.95)
ACCENT = "#1F6F5C"      # teal ink - warna utama chart
ACCENT2 = "#C4622D"     # burnt orange - warna kontras/highlight
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["figure.facecolor"] = "white"

ALPHA = 0.05  # tingkat signifikansi standar


def _is_finite_number(value) -> bool:
    """True jika nilai bisa dipakai sebagai angka hasil uji statistik."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fig_to_base64(fig) -> str:
    """Konversi matplotlib figure ke string base64 PNG untuk ditanam di HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


class EDAAgent:
    """Agent utama. Inisialisasi dengan path file, lalu panggil run_full_analysis()."""

    MAX_GROUP_CARDINALITY = 12   # kolom kategorikal dengan >12 kategori dilewati untuk uji grup
    MAX_NUMERIC_FOR_CORR = 25    # batas kolom numerik yang dimasukkan ke correlation heatmap

    def __init__(self, filepath: str, sheet_name=0):
        self.filepath = filepath
        self.sheet_name = sheet_name
        self.df = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.results = {
            "profile": {},
            "patterns": {},
            "tests": [],
            "charts": {},
            "narrative": [],
            "warnings": [],
        }

    # ------------------------------------------------------------------ #
    # 1. LOAD & PROFILE
    # ------------------------------------------------------------------ #
    def load_data(self):
        if self.filepath.lower().endswith((".xlsx", ".xls")):
            self.df = pd.read_excel(self.filepath, sheet_name=self.sheet_name)
        else:
            # coba beberapa delimiter umum
            try:
                self.df = pd.read_csv(self.filepath)
            except Exception:
                self.df = pd.read_csv(self.filepath, sep=None, engine="python")

        # bersihkan nama kolom
        self.df.columns = [str(c).strip() for c in self.df.columns]

        # buang kolom kosong total / unnamed index bawaan Excel
        self.df = self.df.loc[:, ~self.df.columns.str.contains("^Unnamed", na=False)]
        return self

    def _detect_column_types(self):
        for col in self.df.columns:
            series = self.df[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                self.datetime_cols.append(col)
                continue
            # coba parse sebagai tanggal jika bertipe object dan namanya mengindikasikan tanggal
            if series.dtype == object:
                sample = series.dropna().astype(str).head(20)
                parsed_ok = 0
                for v in sample:
                    try:
                        pd.to_datetime(v)
                        parsed_ok += 1
                    except Exception:
                        pass
                if len(sample) > 0 and parsed_ok / len(sample) > 0.8 and any(
                    ch in col.lower() for ch in ["date", "tanggal", "waktu", "time"]
                ):
                    self.datetime_cols.append(col)
                    continue

            if pd.api.types.is_numeric_dtype(series):
                # numerik dengan sedikit unique value & tipe int -> kemungkinan kategorikal (misal: kode grup)
                if series.nunique(dropna=True) <= 10 and series.dropna().apply(
                    lambda x: float(x).is_integer()
                ).all():
                    self.categorical_cols.append(col)
                else:
                    self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)

    def profile_data(self):
        self._detect_column_types()
        df = self.df
        n_rows, n_cols = df.shape

        missing = df.isna().sum()
        missing_pct = (missing / n_rows * 100).round(2)

        numeric_summary = {}
        for col in self.numeric_cols:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            numeric_summary[col] = {
                "mean": round(float(s.mean()), 3),
                "median": round(float(s.median()), 3),
                "std": round(float(s.std()), 3),
                "min": round(float(s.min()), 3),
                "max": round(float(s.max()), 3),
                "skew": round(float(s.skew()), 3),
                "kurtosis": round(float(s.kurtosis()), 3),
            }

        categorical_summary = {}
        for col in self.categorical_cols:
            vc = df[col].value_counts(dropna=True).head(5)
            categorical_summary[col] = {
                "n_unique": int(df[col].nunique(dropna=True)),
                "top_values": {str(k): int(v) for k, v in vc.items()},
            }

        self.results["profile"] = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "columns": list(df.columns),
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "datetime_cols": self.datetime_cols,
            "missing": {c: {"count": int(missing[c]), "pct": float(missing_pct[c])}
                        for c in df.columns if missing[c] > 0},
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary,
        }
        return self

    # ------------------------------------------------------------------ #
    # 2. PATTERN DETECTION
    # ------------------------------------------------------------------ #
    def detect_patterns(self):
        df = self.df
        patterns = {"outliers": {}, "strong_correlations": [], "high_missing_cols": [],
                    "skewed_cols": [], "constant_cols": []}

        # outlier via IQR
        for col in self.numeric_cols:
            s = df[col].dropna()
            if len(s) < 5:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_outliers = int(((s < lower) | (s > upper)).sum())
            if n_outliers > 0:
                patterns["outliers"][col] = {
                    "count": n_outliers,
                    "pct": round(n_outliers / len(s) * 100, 2),
                }

        # kolom dengan missing tinggi
        for col, info in self.results["profile"]["missing"].items():
            if info["pct"] > 30:
                patterns["high_missing_cols"].append(col)

        # skewness tinggi
        for col, info in self.results["profile"]["numeric_summary"].items():
            if abs(info["skew"]) > 1:
                patterns["skewed_cols"].append({"col": col, "skew": info["skew"]})

        # kolom konstan (tidak informatif)
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 1:
                patterns["constant_cols"].append(col)

        # korelasi kuat antar numerik
        numeric_for_corr = self.numeric_cols[: self.MAX_NUMERIC_FOR_CORR]
        if len(numeric_for_corr) >= 2:
            corr_matrix = df[numeric_for_corr].corr(method="pearson")
            seen = set()
            for c1, c2 in combinations(numeric_for_corr, 2):
                r = corr_matrix.loc[c1, c2]
                if pd.notna(r) and abs(r) >= 0.5 and (c1, c2) not in seen:
                    seen.add((c1, c2))
                    patterns["strong_correlations"].append({
                        "col1": c1, "col2": c2, "r": round(float(r), 3)
                    })
            patterns["strong_correlations"].sort(key=lambda x: -abs(x["r"]))

        self.results["patterns"] = patterns
        return self

    # ------------------------------------------------------------------ #
    # 3. AUTO STATISTICAL TEST SELECTION
    # ------------------------------------------------------------------ #
    def _check_normality(self, sample):
        """Shapiro-Wilk. Untuk n besar, sampling agar tetap valid & cepat."""
        sample = np.asarray(sample)
        if len(sample) < 3:
            return None
        test_sample = sample if len(sample) <= 5000 else np.random.choice(sample, 5000, replace=False)
        try:
            _, p = stats.shapiro(test_sample)
            return p
        except Exception:
            return None

    def run_statistical_tests(self):
        """
        Strategi otomatis:
        - Kategorikal (2 grup) x Numerik -> uji normalitas & varians ->
              t-test independen / Welch t-test / Mann-Whitney U
        - Kategorikal (3-N grup) x Numerik -> uji normalitas & varians ->
              one-way ANOVA / Kruskal-Wallis
        - Kategorikal x Kategorikal -> Chi-square test of independence
        - Numerik x Numerik (korelasi kuat) -> Pearson correlation test (p-value)
        """
        df = self.df
        tests = []

        cat_cols = [c for c in self.categorical_cols
                    if 2 <= df[c].nunique(dropna=True) <= self.MAX_GROUP_CARDINALITY]

        # --- Kategorikal vs Numerik ---
        for cat in cat_cols:
            for num in self.numeric_cols:
                sub = df[[cat, num]].dropna()
                if sub[cat].nunique() < 2 or len(sub) < 6:
                    continue
                grouped = [g[num].values for _, g in sub.groupby(cat) if len(g) >= 2]
                if len(grouped) < 2:
                    continue
                n_groups = len(grouped)

                normal_flags = [self._check_normality(g) for g in grouped]
                is_normal = all(p is not None and p > ALPHA for p in normal_flags)
                try:
                    _, levene_p = stats.levene(*grouped)
                    equal_var = levene_p > ALPHA
                except Exception:
                    equal_var = True

                if n_groups == 2:
                    g1, g2 = grouped[0], grouped[1]
                    if is_normal and equal_var:
                        stat, p = stats.ttest_ind(g1, g2, equal_var=True)
                        method = "Independent t-test"
                    elif is_normal and not equal_var:
                        stat, p = stats.ttest_ind(g1, g2, equal_var=False)
                        method = "Welch's t-test"
                    else:
                        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                        method = "Mann-Whitney U test"
                else:
                    if is_normal and equal_var:
                        stat, p = stats.f_oneway(*grouped)
                        method = "One-way ANOVA"
                    else:
                        stat, p = stats.kruskal(*grouped)
                        method = "Kruskal-Wallis test"

                if not (_is_finite_number(stat) and _is_finite_number(p)):
                    continue

                significant = bool(p < ALPHA)
                tests.append({
                    "type": "group_comparison",
                    "method": method,
                    "variables": f"{num} berdasarkan {cat}",
                    "cat_col": cat,
                    "num_col": num,
                    "n_groups": n_groups,
                    "statistic": round(float(stat), 4),
                    "p_value": float(p),
                    "significant": significant,
                    "interpretation": self._interpret_group_test(
                        method, cat, num, p, significant
                    ),
                })

        # --- Kategorikal vs Kategorikal (Chi-square) ---
        for c1, c2 in combinations(cat_cols, 2):
            sub = df[[c1, c2]].dropna()
            if len(sub) < 6:
                continue
            contingency = pd.crosstab(sub[c1], sub[c2])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue
            try:
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
            except Exception:
                continue
            if not (_is_finite_number(chi2) and _is_finite_number(p)):
                continue
            significant = bool(p < ALPHA)
            tests.append({
                "type": "association",
                "method": "Chi-square test of independence",
                "variables": f"{c1} vs {c2}",
                "cat_col": c1,
                "num_col": c2,
                "n_groups": None,
                "statistic": round(float(chi2), 4),
                "p_value": float(p),
                "significant": significant,
                "interpretation": self._interpret_chi2(c1, c2, p, significant),
            })

        # --- Korelasi kuat -> uji signifikansi Pearson ---
        for pair in self.results["patterns"]["strong_correlations"][:15]:
            c1, c2 = pair["col1"], pair["col2"]
            sub = df[[c1, c2]].dropna()
            if len(sub) < 4:
                continue
            r, p = stats.pearsonr(sub[c1], sub[c2])
            if not (_is_finite_number(r) and _is_finite_number(p)):
                continue
            significant = bool(p < ALPHA)
            tests.append({
                "type": "correlation",
                "method": "Pearson correlation",
                "variables": f"{c1} vs {c2}",
                "cat_col": c1,
                "num_col": c2,
                "n_groups": None,
                "statistic": round(float(r), 4),
                "p_value": float(p),
                "significant": significant,
                "interpretation": self._interpret_correlation(c1, c2, r, p, significant),
            })

        tests.sort(key=lambda t: t["p_value"])
        self.results["tests"] = tests
        return self

    @staticmethod
    def _interpret_group_test(method, cat, num, p, significant):
        if significant:
            return (f"{method} menunjukkan perbedaan rata-rata '{num}' yang signifikan secara "
                    f"statistik antar kategori '{cat}' (p={p:.4f} < 0.05). Kategori '{cat}' "
                    f"kemungkinan besar berpengaruh terhadap nilai '{num}'.")
        return (f"{method} tidak menemukan perbedaan '{num}' yang signifikan antar kategori "
                f"'{cat}' (p={p:.4f} ≥ 0.05). Belum cukup bukti bahwa '{cat}' memengaruhi '{num}'.")

    @staticmethod
    def _interpret_chi2(c1, c2, p, significant):
        if significant:
            return (f"Chi-square test mengindikasikan adanya asosiasi/ketergantungan signifikan "
                    f"antara '{c1}' dan '{c2}' (p={p:.4f} < 0.05).")
        return (f"Tidak ditemukan asosiasi signifikan antara '{c1}' dan '{c2}' "
                f"(p={p:.4f} ≥ 0.05); keduanya tampak independen.")

    @staticmethod
    def _interpret_correlation(c1, c2, r, p, significant):
        strength = "kuat" if abs(r) >= 0.7 else "sedang"
        arah = "positif" if r > 0 else "negatif"
        if significant:
            return (f"Terdapat korelasi {arah} {strength} yang signifikan antara '{c1}' dan "
                    f"'{c2}' (r={r:.3f}, p={p:.4f} < 0.05).")
        return (f"Korelasi antara '{c1}' dan '{c2}' terlihat {strength} (r={r:.3f}) namun "
                f"tidak signifikan secara statistik (p={p:.4f} ≥ 0.05).")

    # ------------------------------------------------------------------ #
    # 4. VISUALIZATION
    # ------------------------------------------------------------------ #
    def generate_visualizations(self):
        charts = {}
        df = self.df

        # --- Missing value overview ---
        missing_info = self.results["profile"]["missing"]
        if missing_info:
            fig, ax = plt.subplots(figsize=(7, max(2, len(missing_info) * 0.4)))
            cols = list(missing_info.keys())
            pcts = [missing_info[c]["pct"] for c in cols]
            ax.barh(cols, pcts, color=ACCENT2)
            ax.set_xlabel("% Missing")
            ax.set_title("Kolom dengan Data Hilang")
            charts["missing"] = _fig_to_base64(fig)

        # --- Correlation heatmap ---
        numeric_for_corr = self.numeric_cols[: self.MAX_NUMERIC_FOR_CORR]
        if len(numeric_for_corr) >= 2:
            corr = df[numeric_for_corr].corr()
            fig, ax = plt.subplots(figsize=(max(5, len(numeric_for_corr) * 0.7),
                                             max(4, len(numeric_for_corr) * 0.6)))
            sns.heatmap(corr, annot=len(numeric_for_corr) <= 12, fmt=".2f", cmap="RdBu_r",
                        center=0, ax=ax, cbar_kws={"label": "r"}, vmin=-1, vmax=1)
            ax.set_title("Matriks Korelasi (Pearson)")
            charts["correlation"] = _fig_to_base64(fig)

        # --- Distribusi numerik (grid histogram) ---
        num_cols_plot = self.numeric_cols[:12]
        if num_cols_plot:
            n = len(num_cols_plot)
            ncols = 3
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows))
            axes = np.array(axes).reshape(-1)
            for i, col in enumerate(num_cols_plot):
                sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color=ACCENT)
                axes[i].set_title(col, fontsize=10)
            for j in range(len(num_cols_plot), len(axes)):
                axes[j].axis("off")
            fig.suptitle("Distribusi Variabel Numerik", y=1.02)
            fig.tight_layout()
            charts["distributions"] = _fig_to_base64(fig)

        # --- Bar chart kategorikal ---
        cat_cols_plot = self.categorical_cols[:8]
        if cat_cols_plot:
            n = len(cat_cols_plot)
            ncols = 2
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
            axes = np.array(axes).reshape(-1)
            for i, col in enumerate(cat_cols_plot):
                vc = df[col].value_counts(dropna=True).head(10)
                axes[i].bar(vc.index.astype(str), vc.values, color=ACCENT2)
                axes[i].set_title(col, fontsize=10)
                axes[i].tick_params(axis="x", rotation=45, labelsize=8)
            for j in range(len(cat_cols_plot), len(axes)):
                axes[j].axis("off")
            fig.suptitle("Frekuensi Variabel Kategorikal", y=1.02)
            fig.tight_layout()
            charts["categorical"] = _fig_to_base64(fig)

        # --- Boxplot untuk uji grup yang signifikan (top 6) ---
        sig_group_tests = [t for t in self.results["tests"]
                            if t["type"] == "group_comparison" and t["significant"]][:6]
        if sig_group_tests:
            n = len(sig_group_tests)
            ncols = 2
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows))
            axes = np.array(axes).reshape(-1)
            for i, t in enumerate(sig_group_tests):
                sub = df[[t["cat_col"], t["num_col"]]].dropna()
                sns.boxplot(data=sub, x=t["cat_col"], y=t["num_col"], ax=axes[i],
                            palette="Set2")
                axes[i].set_title(f"{t['num_col']} vs {t['cat_col']} (p={t['p_value']:.4f})",
                                   fontsize=9)
                axes[i].tick_params(axis="x", rotation=30, labelsize=8)
            for j in range(len(sig_group_tests), len(axes)):
                axes[j].axis("off")
            fig.suptitle("Perbandingan Grup Signifikan", y=1.02)
            fig.tight_layout()
            charts["significant_groups"] = _fig_to_base64(fig)

        # --- Scatter untuk korelasi kuat (top 4) ---
        top_corr = self.results["patterns"]["strong_correlations"][:4]
        if top_corr:
            n = len(top_corr)
            ncols = 2
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            axes = np.array(axes).reshape(-1)
            for i, pair in enumerate(top_corr):
                sub = df[[pair["col1"], pair["col2"]]].dropna()
                axes[i].scatter(sub[pair["col1"]], sub[pair["col2"]], alpha=0.5,
                                 color=ACCENT, edgecolor="white", linewidth=0.3)
                axes[i].set_xlabel(pair["col1"])
                axes[i].set_ylabel(pair["col2"])
                axes[i].set_title(f"r = {pair['r']}", fontsize=10)
            for j in range(len(top_corr), len(axes)):
                axes[j].axis("off")
            fig.suptitle("Hubungan Antar Variabel Berkorelasi Kuat", y=1.02)
            fig.tight_layout()
            charts["scatter_corr"] = _fig_to_base64(fig)

        self.results["charts"] = charts
        return self

    # ------------------------------------------------------------------ #
    # 5. NARRATIVE SUMMARY
    # ------------------------------------------------------------------ #
    def generate_narrative(self):
        p = self.results["profile"]
        pat = self.results["patterns"]
        narrative = []

        narrative.append(
            f"Dataset berisi {p['n_rows']} baris dan {p['n_cols']} kolom "
            f"({len(p['numeric_cols'])} numerik, {len(p['categorical_cols'])} kategorikal"
            + (f", {len(p['datetime_cols'])} tanggal/waktu" if p['datetime_cols'] else "") + ")."
        )

        if p["duplicate_rows"] > 0:
            narrative.append(f"Ditemukan {p['duplicate_rows']} baris duplikat yang sebaiknya diperiksa.")

        if p["missing"]:
            worst = max(p["missing"].items(), key=lambda x: x[1]["pct"])
            narrative.append(
                f"Terdapat {len(p['missing'])} kolom dengan data hilang; paling parah adalah "
                f"'{worst[0]}' ({worst[1]['pct']}% hilang)."
            )
        else:
            narrative.append("Tidak ada data hilang terdeteksi pada dataset ini.")

        if pat["constant_cols"]:
            narrative.append(
                f"Kolom {pat['constant_cols']} bernilai konstan dan tidak memberi informasi "
                f"analitis — pertimbangkan untuk dihapus."
            )

        if pat["outliers"]:
            top_outlier = sorted(pat["outliers"].items(), key=lambda x: -x[1]["pct"])[0]
            narrative.append(
                f"Outlier terdeteksi pada {len(pat['outliers'])} kolom numerik; kolom "
                f"'{top_outlier[0]}' memiliki proporsi outlier tertinggi ({top_outlier[1]['pct']}%)."
            )

        if pat["skewed_cols"]:
            names = ", ".join(f"{s['col']} (skew={s['skew']})" for s in pat["skewed_cols"][:5])
            narrative.append(f"Distribusi tidak simetris (skewed) terdeteksi pada: {names}.")

        if pat["strong_correlations"]:
            top = pat["strong_correlations"][0]
            narrative.append(
                f"Korelasi terkuat ditemukan antara '{top['col1']}' dan '{top['col2']}' "
                f"(r={top['r']})."
            )

        sig_tests = [t for t in self.results["tests"] if t["significant"]]
        if sig_tests:
            narrative.append(
                f"Dari {len(self.results['tests'])} uji statistik yang dijalankan otomatis, "
                f"{len(sig_tests)} di antaranya menunjukkan hasil signifikan (p < 0.05)."
            )
        elif self.results["tests"]:
            narrative.append(
                f"Dari {len(self.results['tests'])} uji statistik yang dijalankan, tidak ada "
                f"yang mencapai signifikansi statistik pada α=0.05."
            )
        else:
            narrative.append(
                "Tidak cukup kombinasi kolom kategorikal/numerik yang memenuhi syarat untuk "
                "uji statistik otomatis."
            )

        self.results["narrative"] = narrative
        return self

    # ------------------------------------------------------------------ #
    # 6. CUSTOM / MANUAL TEST (dipicu manual dropdown ATAU hasil parsing prompt)
    # ------------------------------------------------------------------ #
    def run_custom_test(self, spec: dict) -> dict:
        """
        Jalankan SATU analisis atau uji statistik spesifik sesuai `spec`.
        Mendukung uji inferensial klasik maupun pertanyaan analisis data:
        - ttest, welch, mannwhitney, anova, kruskal, auto (komparasi grup)
        - chi2 (asosiasi kategorikal)
        - pearson, spearman (korelasi numerik)
        - numeric_compare (komparasi 2 kolom numerik)
        - column_stats (statistik deskriptif 1 kolom atau seluruh kolom)
        - distribution, normality (uji sebaran & normalitas Shapiro-Wilk)
        - category_stats (frekuensi & proporsi kategori)
        - dataset_overview (dimensi & struktur dataset)
        - missing_values (analisis data kosong)
        - outlier_report (deteksi pencilan IQR)
        - top_correlations (daftar korelasi terkuat)
        - dataset_summary (ringkasan eksekutif dataset)
        """
        if self.df is None:
            return {"error": "Dataset belum dimuat."}

        df = self.df
        test = (spec.get("test") or "auto").lower()
        col1 = spec.get("col1")
        col2 = spec.get("col2")

        try:
            # 1. Analisis dataset-wide (tidak wajib kolom)
            if test == "dataset_overview":
                return self._run_custom_dataset_overview()
            elif test == "missing_values":
                return self._run_custom_missing_values(col1)
            elif test == "outlier_report":
                return self._run_custom_outlier_report(col1)
            elif test == "top_correlations":
                return self._run_custom_top_correlations()
            elif test == "dataset_summary":
                return self._run_custom_dataset_summary()

            # 2. Analisis satu kolom
            if test in ("column_stats",):
                if col1 == "all" or not col1:
                    return self._run_custom_column_stats("all", stat_focus=col2 or "all")
                if col1 not in df.columns:
                    return {"error": f"Kolom '{col1}' tidak ditemukan di dataset."}
                return self._run_custom_column_stats(col1, stat_focus=col2 or "all")

            if test in ("distribution", "normality"):
                target_col = col1 or (self.numeric_cols[0] if self.numeric_cols else None)
                if not target_col or target_col not in df.columns:
                    return {"error": f"Kolom '{target_col}' tidak ditemukan di dataset."}
                return self._run_custom_distribution(target_col)

            if test == "category_stats":
                target_col = col1 or (self.categorical_cols[0] if self.categorical_cols else None)
                if not target_col or target_col not in df.columns:
                    return {"error": f"Kolom '{target_col}' tidak ditemukan di dataset."}
                return self._run_custom_category_stats(target_col)

            # 3. Komparasi 2 kolom numerik
            if test == "numeric_compare":
                if not col1 or col1 not in df.columns:
                    return {"error": f"Kolom '{col1}' tidak ditemukan di dataset."}
                if not col2 or col2 not in df.columns:
                    return {"error": f"Kolom '{col2}' tidak ditemukan di dataset."}
                return self._run_custom_numeric_compare(col1, col2)

            # 4. Validasi kolom untuk uji 2 variabel
            if not col1 or col1 not in df.columns:
                return {"error": f"Kolom '{col1}' tidak ditemukan di dataset."}
            if not col2 or col2 not in df.columns:
                return {"error": f"Kolom '{col2}' tidak ditemukan di dataset."}

            # Otomatis deteksi jenis uji jika 'auto'
            if test == "auto":
                if col1 in self.numeric_cols and col2 in self.numeric_cols:
                    return self._run_custom_numeric_compare(col1, col2)
                elif col1 in self.categorical_cols and col2 in self.categorical_cols:
                    return self._run_custom_chi2(col1, col2)
                else:
                    return self._run_custom_group_test(test, col1, col2)

            if test in ("ttest", "welch", "mannwhitney", "anova", "kruskal"):
                return self._run_custom_group_test(test, col1, col2)
            elif test == "chi2":
                return self._run_custom_chi2(col1, col2)
            elif test in ("pearson", "spearman"):
                return self._run_custom_correlation(test, col1, col2)
            else:
                return {"error": f"Jenis analisis '{test}' tidak dikenali."}
        except Exception as e:
            return {"error": f"Gagal menjalankan analisis: {e}"}

    def _run_custom_group_test(self, test, cat_col, num_col):
        # Auto-swap jika pengguna terbalik menulis (misal: Fresh berdasarkan Channel)
        if cat_col in self.numeric_cols and num_col in self.categorical_cols:
            cat_col, num_col = num_col, cat_col

        df = self.df
        sub = df[[cat_col, num_col]].dropna()
        if not pd.api.types.is_numeric_dtype(sub[num_col]):
            return {"error": f"Kolom '{num_col}' harus numerik untuk uji perbandingan grup."}

        grouped_dict = {k: g[num_col].values for k, g in sub.groupby(cat_col) if len(g) >= 2}
        if len(grouped_dict) < 2:
            return {"error": f"Kolom '{cat_col}' butuh minimal 2 grup dengan masing-masing ≥2 data."}
        group_names = list(grouped_dict.keys())
        grouped = list(grouped_dict.values())
        n_groups = len(grouped)

        normal_flags = [self._check_normality(g) for g in grouped]
        is_normal = all(p is not None and p > ALPHA for p in normal_flags)
        try:
            _, levene_p = stats.levene(*grouped)
            equal_var = levene_p > ALPHA
        except Exception:
            equal_var = True

        # Jika user memaksa metode tertentu, pakai itu; jika 'auto', pilih otomatis
        if test == "auto":
            if n_groups == 2:
                method_key = "ttest" if (is_normal and equal_var) else (
                    "welch" if is_normal else "mannwhitney")
            else:
                method_key = "anova" if (is_normal and equal_var) else "kruskal"
        else:
            method_key = test

        if method_key == "ttest":
            if n_groups != 2:
                return {"error": "Independent t-test hanya berlaku untuk tepat 2 grup. Gunakan ANOVA/Kruskal untuk >2 grup."}
            stat, p = stats.ttest_ind(grouped[0], grouped[1], equal_var=True)
            method_name = "Independent t-test"
        elif method_key == "welch":
            if n_groups != 2:
                return {"error": "Welch's t-test hanya berlaku untuk tepat 2 grup."}
            stat, p = stats.ttest_ind(grouped[0], grouped[1], equal_var=False)
            method_name = "Welch's t-test"
        elif method_key == "mannwhitney":
            if n_groups != 2:
                return {"error": "Mann-Whitney U hanya berlaku untuk tepat 2 grup."}
            stat, p = stats.mannwhitneyu(grouped[0], grouped[1], alternative="two-sided")
            method_name = "Mann-Whitney U test"
        elif method_key == "anova":
            stat, p = stats.f_oneway(*grouped)
            method_name = "One-way ANOVA"
        elif method_key == "kruskal":
            stat, p = stats.kruskal(*grouped)
            method_name = "Kruskal-Wallis test"
        else:
            return {"error": f"Metode '{method_key}' tidak dikenali."}

        if not (_is_finite_number(stat) and _is_finite_number(p)):
            return {"error": "Uji tidak menghasilkan statistik valid. Periksa apakah data konstan atau variasinya terlalu rendah."}

        significant = bool(p < ALPHA)
        assumption_note = (
            f"Cek asumsi: normalitas {'terpenuhi' if is_normal else 'TIDAK terpenuhi'} "
            f"(Shapiro-Wilk), homogenitas varians {'terpenuhi' if equal_var else 'TIDAK terpenuhi'} (Levene)."
        )

        fig, ax = plt.subplots(figsize=(6, 4.2))
        sns.boxplot(data=sub, x=cat_col, y=num_col, ax=ax, palette="Set2")
        ax.set_title(f"{num_col} berdasarkan {cat_col}\n{method_name} (p={p:.4f})", fontsize=11)
        fig.tight_layout()
        chart = _fig_to_base64(fig)

        return {
            "method": method_name,
            "variables": f"{num_col} berdasarkan {cat_col}",
            "statistic": round(float(stat), 4),
            "p_value": float(p),
            "significant": significant,
            "interpretation": self._interpret_group_test(method_name, cat_col, num_col, p, significant),
            "assumption_note": assumption_note,
            "chart": chart,
        }

    def _run_custom_chi2(self, col1, col2):
        df = self.df
        sub = df[[col1, col2]].dropna()
        contingency = pd.crosstab(sub[col1], sub[col2])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return {"error": f"'{col1}' dan '{col2}' butuh masing-masing minimal 2 kategori unik."}
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        if not (_is_finite_number(chi2) and _is_finite_number(p)):
            return {"error": "Chi-square tidak menghasilkan statistik valid. Periksa apakah tabel kontingensi terlalu jarang atau konstan."}
        significant = bool(p < ALPHA)

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        contingency.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
        ax.set_title(f"{col1} vs {col2}\nChi-square test (p={p:.4f})", fontsize=11)
        ax.legend(title=col2, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        chart = _fig_to_base64(fig)

        return {
            "method": "Chi-square test of independence",
            "variables": f"{col1} vs {col2}",
            "statistic": round(float(chi2), 4),
            "p_value": float(p),
            "significant": significant,
            "interpretation": self._interpret_chi2(col1, col2, p, significant),
            "assumption_note": f"Derajat kebebasan (dof) = {dof}.",
            "chart": chart,
        }

    def _run_custom_correlation(self, test, col1, col2):
        df = self.df
        sub = df[[col1, col2]].dropna()
        if not (pd.api.types.is_numeric_dtype(sub[col1]) and pd.api.types.is_numeric_dtype(sub[col2])):
            return {"error": f"'{col1}' dan '{col2}' harus keduanya numerik untuk uji korelasi."}
        if len(sub) < 4:
            return {"error": "Data valid (non-missing) untuk kedua kolom kurang dari 4 baris."}

        if test == "pearson":
            r, p = stats.pearsonr(sub[col1], sub[col2])
            method_name = "Pearson correlation"
        else:
            r, p = stats.spearmanr(sub[col1], sub[col2])
            method_name = "Spearman correlation"
        if not (_is_finite_number(r) and _is_finite_number(p)):
            return {"error": "Uji korelasi tidak menghasilkan statistik valid. Salah satu kolom mungkin konstan atau tidak memiliki variasi cukup."}
        significant = bool(p < ALPHA)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(sub[col1], sub[col2], alpha=0.55, color=ACCENT, edgecolor="white", linewidth=0.3)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f"{method_name}\nr={r:.3f}, p={p:.4f}", fontsize=11)
        fig.tight_layout()
        chart = _fig_to_base64(fig)

        return {
            "method": method_name,
            "variables": f"{col1} vs {col2}",
            "statistic": round(float(r), 4),
            "p_value": float(p),
            "significant": significant,
            "interpretation": self._interpret_correlation(col1, col2, r, p, significant),
            "assumption_note": "Spearman dipakai jika hubungan tidak linear/data tidak normal." if test == "spearman" else "Pearson mengasumsikan hubungan linear.",
            "chart": chart,
        }

    def _run_custom_column_stats(self, col: str, stat_focus: str = "all") -> dict:
        df = self.df
        if col == "all":
            # Ringkasan seluruh kolom numerik
            summary_data = []
            for c in self.numeric_cols:
                sc = df[c].dropna()
                if len(sc) > 0:
                    summary_data.append({
                        "kolom": c,
                        "mean": sc.mean(),
                        "median": sc.median(),
                        "min": sc.min(),
                        "max": sc.max(),
                        "std": sc.std(),
                    })
            if not summary_data:
                return {"error": "Tidak ada kolom numerik yang dapat dihitung."}

            sdf = pd.DataFrame(summary_data).sort_values(by="mean", ascending=False)
            highest_mean_col = sdf.iloc[0]["kolom"]
            lowest_mean_col = sdf.iloc[-1]["kolom"]

            fig, ax = plt.subplots(figsize=(6.5, max(3.5, len(sdf) * 0.45)))
            sns.barplot(data=sdf, y="kolom", x="mean", ax=ax, palette="mako")
            ax.set_title("Perbandingan Rata-rata (Mean) Kolom Numerik", fontsize=11)
            ax.set_xlabel("Rata-rata")
            ax.set_ylabel("")
            fig.tight_layout()
            chart = _fig_to_base64(fig)

            metrics = [
                {"label": "Total Kolom Numerik", "value": str(len(sdf))},
                {"label": "Rata-rata Tertinggi", "value": f"{highest_mean_col} ({sdf.iloc[0]['mean']:,.2f})"},
                {"label": "Rata-rata Terendah", "value": f"{lowest_mean_col} ({sdf.iloc[-1]['mean']:,.2f})"},
            ]
            interpretation = (
                f"Dari {len(sdf)} kolom numerik, nilai rata-rata tertinggi berada pada kolom **{highest_mean_col}** "
                f"({sdf.iloc[0]['mean']:,.2f}), sedangkan yang terendah adalah **{lowest_mean_col}** "
                f"({sdf.iloc[-1]['mean']:,.2f})."
            )
            return {
                "method": "Statistik Deskriptif — Seluruh Kolom Numerik",
                "variables": "Semua kolom numerik",
                "statistic": None,
                "p_value": None,
                "significant": None,
                "metrics": metrics,
                "interpretation": interpretation,
                "assumption_note": "Perhitungan statistik deskriptif dihitung dari baris data valid non-missing.",
                "chart": chart,
            }

        s = df[col].dropna()
        if len(s) == 0:
            return {"error": f"Kolom '{col}' tidak memiliki data valid."}

        mean_val = float(s.mean())
        median_val = float(s.median())
        min_val = float(s.min())
        max_val = float(s.max())
        sum_val = float(s.sum())
        std_val = float(s.std())
        skew_val = float(s.skew())

        # Subplots: Boxplot di atas, Histogram + KDE di bawah
        fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(6.5, 5), gridspec_kw={"height_ratios": [0.3, 0.7]}, sharex=True)
        sns.boxplot(x=s, ax=ax_box, color=ACCENT, fliersize=3)
        ax_box.set(xlabel="")
        ax_box.set_title(f"Distribusi & Statistik: {col}", fontsize=11)

        sns.histplot(s, kde=True, ax=ax_hist, color=ACCENT, edgecolor="white", alpha=0.6)
        ax_hist.axvline(mean_val, color="#c0392b", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:,.1f}")
        ax_hist.axvline(median_val, color="#2980b9", linestyle="-.", linewidth=1.5, label=f"Median: {median_val:,.1f}")
        ax_hist.set_xlabel(col)
        ax_hist.set_ylabel("Frekuensi")
        ax_hist.legend(loc="upper right", fontsize=8.5)
        fig.tight_layout()
        chart = _fig_to_base64(fig)

        metrics = [
            {"label": "Rata-rata (Mean)", "value": f"{mean_val:,.2f}"},
            {"label": "Median", "value": f"{median_val:,.2f}"},
            {"label": "Standar Deviasi", "value": f"{std_val:,.2f}"},
            {"label": "Nilai Terendah (Min)", "value": f"{min_val:,.2f}"},
            {"label": "Nilai Tertinggi (Max)", "value": f"{max_val:,.2f}"},
            {"label": "Total (Sum)", "value": f"{sum_val:,.2f}"},
        ]

        skew_desc = (
            "miring ke kanan (positively skewed) karena mean > median" if skew_val > 0.5 else (
                "miring ke kiri (negatively skewed) karena mean < median" if skew_val < -0.5 else "relatif simetris"
            )
        )
        interpretation = (
            f"Kolom **{col}** memiliki nilai rata-rata **{mean_val:,.2f}** dengan median **{median_val:,.2f}**. "
            f"Rentang data bergerak dari nilai minimum **{min_val:,.2f}** hingga maksimum **{max_val:,.2f}** "
            f"(standar deviasi: {std_val:,.2f}). Distribusi data tergolong {skew_desc} (skewness = {skew_val:.2f})."
        )
        return {
            "method": f"Statistik Deskriptif — {col}",
            "variables": col,
            "statistic": round(mean_val, 4),
            "p_value": None,
            "significant": None,
            "metrics": metrics,
            "interpretation": interpretation,
            "assumption_note": f"Dihitung dari {len(s):,} data valid ({df[col].isna().sum():,} nilai kosong).",
            "chart": chart,
        }

    def _run_custom_distribution(self, col: str) -> dict:
        df = self.df
        s = df[col].dropna()
        if len(s) < 4:
            return {"error": f"Data valid pada kolom '{col}' kurang dari 4 baris."}

        # Uji Shapiro-Wilk (maksimal 5000 sampel untuk stabilitas)
        sample_s = s.sample(min(len(s), 5000), random_state=42)
        stat, p = stats.shapiro(sample_s)
        skew_val = float(s.skew())
        kurt_val = float(s.kurtosis())
        is_normal = bool(p > ALPHA and abs(skew_val) < 1.0)

        fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(6.5, 5), gridspec_kw={"height_ratios": [0.3, 0.7]}, sharex=True)
        sns.boxplot(x=s, ax=ax_box, color=ACCENT2, fliersize=3)
        ax_box.set(xlabel="")
        ax_box.set_title(f"Uji Bentuk Sebaran & Normalitas: {col}\nShapiro-Wilk W={stat:.4f} (p={p:.4e})", fontsize=11)

        sns.histplot(s, kde=True, ax=ax_hist, color=ACCENT2, edgecolor="white", alpha=0.6)
        ax_hist.set_xlabel(col)
        ax_hist.set_ylabel("Frekuensi")
        fig.tight_layout()
        chart = _fig_to_base64(fig)

        metrics = [
            {"label": "Shapiro-Wilk W", "value": f"{stat:.4f}"},
            {"label": "p-value", "value": f"{p:.4e}"},
            {"label": "Status Normalitas", "value": "Normal" if is_normal else "Tidak Normal"},
            {"label": "Skewness", "value": f"{skew_val:.2f}"},
            {"label": "Kurtosis", "value": f"{kurt_val:.2f}"},
        ]

        if is_normal:
            interpretation = (
                f"Berdasarkan uji Shapiro-Wilk (p = {p:.4e} > 0.05) dan skewness ({skew_val:.2f}), "
                f"kolom **{col}** berdistribusi normal. Asumsi parametrik terpenuhi sehingga uji seperti "
                f"t-test, ANOVA, atau Pearson correlation cocok digunakan."
            )
        else:
            skew_label = "condong ke kanan (right-skewed)" if skew_val > 0.5 else ("condong ke kiri (left-skewed)" if skew_val < -0.5 else "tidak simetris")
            interpretation = (
                f"Uji Shapiro-Wilk menghasilkan p-value = {p:.4e} (< 0.05), yang menunjukkan bahwa kolom **{col}** "
                f"secara statistik **TIDAK berdistribusi normal**. Sebaran data {skew_label} dengan kurtosis {kurt_val:.2f}. "
                f"Direkomendasikan menggunakan uji non-parametrik (Mann-Whitney, Kruskal-Wallis, Spearman) atau transformasi data."
            )

        return {
            "method": f"Uji Normalitas & Distribusi — {col}",
            "variables": col,
            "statistic": round(float(stat), 4),
            "p_value": float(p),
            "significant": bool(p < ALPHA),
            "metrics": metrics,
            "interpretation": interpretation,
            "assumption_note": "Uji normalitas Shapiro-Wilk menguji hipotesis nol (H0) bahwa data berasal dari populasi berdistribusi normal.",
            "chart": chart,
        }

    def _run_custom_category_stats(self, col: str) -> dict:
        df = self.df
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            return {"error": f"Kolom '{col}' tidak memiliki data valid."}

        vc = s.value_counts()
        total_n = len(s)
        n_unique = len(vc)
        top_name = str(vc.index[0])
        top_count = int(vc.iloc[0])
        top_pct = (top_count / total_n) * 100

        # Bar chart top kategori
        plot_df = vc.head(10).reset_index()
        plot_df.columns = ["kategori", "jumlah"]

        fig, ax = plt.subplots(figsize=(6.5, max(3.5, len(plot_df) * 0.45)))
        sns.barplot(data=plot_df, y="kategori", x="jumlah", ax=ax, palette="Set2")
        for i, val in enumerate(plot_df["jumlah"]):
            ax.text(val + (max(plot_df["jumlah"]) * 0.01), i, f" {val:,} ({val/total_n*100:.1f}%)", va="center", fontsize=8.5)
        ax.set_title(f"Distribusi Frekuensi Kategori: {col}", fontsize=11)
        ax.set_xlabel("Jumlah Sampel")
        ax.set_ylabel("")
        fig.tight_layout()
        chart = _fig_to_base64(fig)

        metrics = [
            {"label": "Kategori Unik", "value": str(n_unique)},
            {"label": "Modus (Terbanyak)", "value": f"{top_name} ({top_count:,})"},
            {"label": "Proporsi Terbanyak", "value": f"{top_pct:.1f}%"},
            {"label": "Total Sampel Valid", "value": f"{total_n:,}"},
        ]
        interpretation = (
            f"Kolom kategorikal **{col}** memiliki **{n_unique}** kategori unik. Kategori paling dominan adalah "
            f"**{top_name}** dengan jumlah {top_count:,} baris ({top_pct:.1f}% dari keseluruhan data)."
        )
        return {
            "method": f"Analisis Kategori — {col}",
            "variables": col,
            "statistic": None,
            "p_value": None,
            "significant": None,
            "metrics": metrics,
            "interpretation": interpretation,
            "assumption_note": f"Dihitung dari {total_n:,} sampel valid ({df[col].isna().sum():,} nilai kosong).",
            "chart": chart,
        }

    def _run_custom_numeric_compare(self, col1: str, col2: str) -> dict:
        df = self.df
        sub = df[[col1, col2]].dropna()
        if len(sub) < 4:
            return {"error": f"Data valid untuk kolom '{col1}' dan '{col2}' kurang dari 4 baris."}

        m1, m2 = float(sub[col1].mean()), float(sub[col2].mean())
        med1, med2 = float(sub[col1].median()), float(sub[col2].median())
        diff = m1 - m2
        r, p_corr = stats.pearsonr(sub[col1], sub[col2])
        t_stat, p_ttest = stats.ttest_ind(sub[col1], sub[col2], equal_var=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4))
        # Plot 1: Boxplot perbandingan
        melted = pd.melt(sub[[col1, col2]], value_vars=[col1, col2], var_name="Kolom", value_name="Nilai")
        sns.boxplot(data=melted, x="Kolom", y="Nilai", ax=ax1, palette=[ACCENT, ACCENT2])
        ax1.set_title(f"Perbandingan Nilai\n(Welch p={p_ttest:.4f})", fontsize=10.5)

        # Plot 2: Scatter plot korelasi
        sns.regplot(data=sub, x=col1, y=col2, ax=ax2, color=ACCENT, scatter_kws={"alpha": 0.5, "s": 20}, line_kws={"color": ACCENT2})
        ax2.set_title(f"Hubungan Korelasi\nr={r:.3f} (p={p_corr:.4f})", fontsize=10.5)
        fig.tight_layout()
        chart = _fig_to_base64(fig)

        metrics = [
            {"label": f"Rata-rata {col1}", "value": f"{m1:,.2f}"},
            {"label": f"Rata-rata {col2}", "value": f"{m2:,.2f}"},
            {"label": "Selisih Rata-rata", "value": f"{diff:,.2f}"},
            {"label": "Korelasi Pearson (r)", "value": f"{r:.3f}"},
            {"label": "Uji Beda (p-value)", "value": f"{p_ttest:.4f}"},
        ]
        sig_diff = "terdapat perbedaan rata-rata yang signifikan" if p_ttest < ALPHA else "tidak terdapat perbedaan rata-rata yang signifikan"
        interpretation = (
            f"Perbandingan antara **{col1}** (rata-rata: {m1:,.2f}, median: {med1:,.2f}) dan **{col2}** "
            f"(rata-rata: {m2:,.2f}, median: {med2:,.2f}) menghasilkan selisih rata-rata sebesar {abs(diff):,.2f}. "
            f"Berdasarkan uji beda independen Welch, {sig_diff} (p = {p_ttest:.4f}). "
            f"Korelasi linear antara kedua variabel bernilai r = {r:.3f} (p = {p_corr:.4f})."
        )
        return {
            "method": f"Komparasi Variabel Numerik — {col1} vs {col2}",
            "variables": f"{col1} vs {col2}",
            "statistic": round(float(t_stat), 4),
            "p_value": float(p_ttest),
            "significant": bool(p_ttest < ALPHA),
            "metrics": metrics,
            "interpretation": interpretation,
            "assumption_note": "Uji komparasi menggunakan Welch's t-test (bebas asumsi kesamaan varians) dan korelasi Pearson.",
            "chart": chart,
        }

    def _run_custom_dataset_overview(self) -> dict:
        df = self.df
        n_rows, n_cols = df.shape
        dup_rows = int(df.duplicated().sum())
        missing_total = int(df.isna().sum().sum())
        missing_cols = int((df.isna().sum() > 0).sum())

        metrics = [
            {"label": "Total Baris", "value": f"{n_rows:,}"},
            {"label": "Total Kolom", "value": str(n_cols)},
            {"label": "Kolom Numerik", "value": str(len(self.numeric_cols))},
            {"label": "Kolom Kategorikal", "value": str(len(self.categorical_cols))},
            {"label": "Baris Duplikat", "value": str(dup_rows)},
            {"label": "Total Nilai Kosong", "value": str(missing_total)},
        ]

        # Chart: ringkasan tipe data
        type_counts = pd.Series({
            "Numerik": len(self.numeric_cols),
            "Kategorikal": len(self.categorical_cols),
            "Tanggal/Waktu": len(self.datetime_cols),
        })
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        sns.barplot(x=type_counts.index, y=type_counts.values, ax=ax, palette=[ACCENT, ACCENT2, "#34495e"])
        ax.set_title("Komposisi Tipe Kolom Dataset", fontsize=11)
        ax.set_ylabel("Jumlah Kolom")
        fig.tight_layout()
        chart = _fig_to_base64(fig)

        interpretation = (
            f"Dataset ini memiliki ukuran **{n_rows:,} baris** dan **{n_cols} kolom** "
            f"({len(self.numeric_cols)} numerik, {len(self.categorical_cols)} kategorikal). "
            f"Terdapat {dup_rows:,} baris duplikat dan {missing_total:,} nilai kosong "
            f"yang tersebar di {missing_cols} kolom."
        )
        return {
            "method": "Ringkasan Dimensi & Profil Dataset",
            "variables": "Keseluruhan Dataset",
            "statistic": None,
            "p_value": None,
            "significant": None,
            "metrics": metrics,
            "interpretation": interpretation,
            "assumption_note": "Tipe data dideteksi secara otomatis berdasarkan karakteristik nilai masing-masing kolom.",
            "chart": chart,
        }

    def _run_custom_missing_values(self, col: str = None) -> dict:
        df = self.df
        missing = df.isna().sum()
        total_rows = len(df)
        missing_df = pd.DataFrame({
            "kolom": missing.index,
            "missing_count": missing.values,
            "missing_pct": (missing.values / total_rows * 100).round(2),
        }).sort_values(by="missing_count", ascending=False)

        problem_cols = missing_df[missing_df["missing_count"] > 0]
        total_missing = int(missing.sum())

        metrics = [
            {"label": "Total Nilai Kosong", "value": f"{total_missing:,}"},
            {"label": "Kolom Ada Missing", "value": str(len(problem_cols))},
            {"label": "Kolom Lengkap (100%)", "value": str(len(df.columns) - len(problem_cols))},
        ]

        if len(problem_cols) == 0:
            interpretation = (
                f"Kondisi kualitas data sangat baik: **tidak ditemukan nilai kosong (0 missing value)** pada seluruh "
                f"{len(df.columns)} kolom dari total {total_rows:,} baris data."
            )
            chart = None
        else:
            fig, ax = plt.subplots(figsize=(6.5, max(3.5, len(problem_cols) * 0.45)))
            sns.barplot(data=problem_cols, y="kolom", x="missing_pct", ax=ax, color=ACCENT2)
            ax.set_title("Persentase Nilai Kosong per Kolom (%)", fontsize=11)
            ax.set_xlabel("Persentase Missing (%)")
            ax.set_ylabel("")
            fig.tight_layout()
            chart = _fig_to_base64(fig)

            top_prob = problem_cols.iloc[0]
            interpretation = (
                f"Ditemukan {total_missing:,} nilai kosong pada **{len(problem_cols)} kolom**. "
                f"Kolom dengan missing value terbanyak adalah **{top_prob['kolom']}** sebanyak "
                f"{int(top_prob['missing_count']):,} baris ({top_prob['missing_pct']}%), diikuti kolom lainnya."
            )

        return {
            "method": "Analisis Kualitas Data & Missing Values",
            "variables": "Pemeriksaan Nilai Kosong",
            "statistic": None,
            "p_value": None,
            "significant": None,
            "metrics": metrics,
            "interpretation": interpretation,
            "assumption_note": "Evaluasi kelengkapan data penting sebelum menjalankan pemodelan prediktif atau inferensi statistik.",
            "chart": chart,
        }

    def _run_custom_outlier_report(self, col: str = None) -> dict:
        df = self.df
        outlier_data = []
        for c in self.numeric_cols:
            s = df[c].dropna()
            if len(s) >= 4:
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    cnt = int(((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum())
                    pct = round(cnt / len(s) * 100, 2)
                    outlier_data.append({"kolom": c, "outlier_count": cnt, "outlier_pct": pct})

        if not outlier_data:
            return {"error": "Tidak ditemukan kolom numerik yang cukup untuk deteksi outlier."}

        odf = pd.DataFrame(outlier_data).sort_values(by="outlier_count", ascending=False)
        top_row = odf.iloc[0]
        total_outliers = int(odf["outlier_count"].sum())

        fig, ax = plt.subplots(figsize=(6.5, max(3.5, len(odf) * 0.45)))
        sns.barplot(data=odf, y="kolom", x="outlier_count", ax=ax, palette="flare")
        ax.set_title("Jumlah Pencilan (Outlier IQR) per Kolom", fontsize=11)
        ax.set_xlabel("Jumlah Pencilan")
        ax.set_ylabel("")
        fig.tight_layout()
        chart = _fig_to_base64(fig)

        metrics = [
            {"label": "Total Outlier Terdeteksi", "value": f"{total_outliers:,}"},
            {"label": "Kolom Terbanyak Outlier", "value": f"{top_row['kolom']} ({int(top_row['outlier_count']):,})"},
            {"label": "Persentase Outlier Tertinggi", "value": f"{top_row['outlier_pct']}%"},
        ]
        interpretation = (
            f"Berdasarkan metode Interquartile Range (IQR 1.5×), kolom **{top_row['kolom']}** memiliki jumlah "
            f"pencilan terbanyak yaitu **{int(top_row['outlier_count']):,} data** ({top_row['outlier_pct']}% dari total baris). "
            f"Pencilan ini dapat memengaruhi estimasi mean dan varians, sehingga pemakaian median atau metode robust dianjurkan."
        )
        return {
            "method": "Deteksi Pencilan (Outlier) IQR",
            "variables": "Kolom Numerik",
            "statistic": None,
            "p_value": None,
            "significant": None,
            "metrics": metrics,
            "interpretation": interpretation,
            "assumption_note": "Pencilan diidentifikasi menggunakan rumus Tukey: Nilai < Q1 - 1.5×IQR atau Nilai > Q3 + 1.5×IQR.",
            "chart": chart,
        }

    def _run_custom_top_correlations(self) -> dict:
        df = self.df
        if len(self.numeric_cols) < 2:
            return {"error": "Dataset membutuhkan minimal 2 kolom numerik untuk analisis korelasi."}

        corr = df[self.numeric_cols].corr()
        pairs = []
        for i, c1 in enumerate(self.numeric_cols):
            for j, c2 in enumerate(self.numeric_cols):
                if i < j:
                    val = corr.loc[c1, c2]
                    if _is_finite_number(val):
                        pairs.append({
                            "pasangan": f"{c1} — {c2}",
                            "r": float(val),
                            "abs_r": abs(float(val)),
                        })

        if not pairs:
            return {"error": "Tidak ditemukan pasangan korelasi yang valid."}

        pdf = pd.DataFrame(pairs).sort_values(by="abs_r", ascending=False).head(8)
        top_pair = pdf.iloc[0]

        fig, ax = plt.subplots(figsize=(6.5, max(3.5, len(pdf) * 0.45)))
        colors = [ACCENT if x >= 0 else ACCENT2 for x in pdf["r"]]
        ax.barh(pdf["pasangan"], pdf["r"], color=colors)
        ax.set_xlim(-1, 1)
        ax.axvline(0, color="#666666", linewidth=0.8, linestyle="--")
        ax.set_title("Pasangan Korelasi Linear Terkuat (Pearson r)", fontsize=11)
        ax.set_xlabel("Koefisien Korelasi (r)")
        fig.tight_layout()
        chart = _fig_to_base64(fig)

        metrics = [
            {"label": "Korelasi Terkuat", "value": f"{top_pair['pasangan']}"},
            {"label": "Nilai r", "value": f"{top_pair['r']:.3f}"},
            {"label": "Arah Hubungan", "value": "Positif Kuat" if top_pair['r'] > 0.6 else ("Negatif Kuat" if top_pair['r'] < -0.6 else "Moderat")},
        ]
        interpretation = (
            f"Korelasi linear paling kuat tercatat antara **{top_pair['pasangan']}** dengan nilai **r = {top_pair['r']:.3f}**. "
            f"Nilai ini mengindikasikan {'hubungan searah yang sangat erat' if top_pair['r'] > 0 else 'hubungan berlawanan arah'} "
            f"antara kedua variabel tersebut."
        )
        return {
            "method": "Peringkat Korelasi Terkuat (Top Correlations)",
            "variables": "Seluruh Pasangan Kolom Numerik",
            "statistic": round(float(top_pair["r"]), 4),
            "p_value": None,
            "significant": None,
            "metrics": metrics,
            "interpretation": interpretation,
            "assumption_note": "Korelasi Pearson mengukur kekuatan hubungan linear antara -1.0 (negatif sempurna) hingga +1.0 (positif sempurna).",
            "chart": chart,
        }

    def _run_custom_dataset_summary(self) -> dict:
        df = self.df
        n_rows, n_cols = df.shape
        dup_rows = int(df.duplicated().sum())
        missing_cnt = int(df.isna().sum().sum())
        top_corr = self.results.get("patterns", {}).get("strong_correlations", [])
        if top_corr:
            c1 = top_corr[0].get("col1") or top_corr[0].get("var1", "")
            c2 = top_corr[0].get("col2") or top_corr[0].get("var2", "")
            top_corr_str = f"{c1} & {c2} (r={top_corr[0]['r']:.2f})"
        else:
            top_corr_str = "-"

        metrics = [
            {"label": "Dimensi Data", "value": f"{n_rows:,} × {n_cols}"},
            {"label": "Kualitas (Missing)", "value": "Bersih (0)" if missing_cnt == 0 else f"{missing_cnt:,} kosong"},
            {"label": "Baris Duplikat", "value": f"{dup_rows:,}"},
            {"label": "Korelasi Terkuat", "value": top_corr_str},
        ]

        interpretation = (
            f"**Ringkasan Eksekutif Dataset:**\n"
            f"1. **Struktur**: Dataset memuat {n_rows:,} baris dan {n_cols} kolom, terdiri dari {len(self.numeric_cols)} variabel numerik dan {len(self.categorical_cols)} variabel kategorikal.\n"
            f"2. **Kebersihan**: {'Data dalam kondisi lengkap tanpa nilai kosong.' if missing_cnt == 0 else f'Terdapat {missing_cnt:,} sel kosong yang perlu penanganan.'} Duplikasi tercatat {dup_rows:,} baris.\n"
            f"3. **Pola Hubungan**: Hubungan antar variabel terkuat ada pada {top_corr_str}.\n"
            f"4. **Rekomendasi**: Gunakan visualisasi dan uji inferensial grup yang telah diidentifikasi untuk penggalian hipotesis lebih mendalam."
        )
        return {
            "method": "Ikhtisar & Kesimpulan Dataset",
            "variables": "Ringkasan Global",
            "statistic": None,
            "p_value": None,
            "significant": None,
            "metrics": metrics,
            "interpretation": interpretation,
            "assumption_note": "Dihasilkan secara otomatis dari profil statistik, deteksi anomali, dan uji inferensial agent.",
            "chart": None,
        }

    # ------------------------------------------------------------------ #
    # ORCHESTRATOR
    # ------------------------------------------------------------------ #
    def run_full_analysis(self):
        (self.load_data()
             .profile_data()
             .detect_patterns()
             .run_statistical_tests()
             .generate_visualizations()
             .generate_narrative())
        return self.results

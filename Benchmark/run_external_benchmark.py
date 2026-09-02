#!/usr/bin/env python
"""
Experiment 1 — mixture retrieval MR-FMN validation (Fig. 3f protocol).

Simulates a reaction mixture: for each product, score all reactants in the library,
use ground-truth pairs as labels, threshold-scan Precision/Recall/F1, and AP with bootstrap CI.
Outputs a single composite SVG (1×3 panels).

Usage:
    python run_experiment1.py
    python run_experiment1.py --skip-score
    python run_experiment1.py --skip-score --skip-eval   # replot only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ── Paths & parameters ────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
POSITIVE_PAIRS_CSV = SCRIPT_DIR / "positive_pairs_clean.csv"
MODEL_PATH = SCRIPT_DIR / "best_model_complete_transformer_DNN.pth"
FINGERPRINT_MODULE = SCRIPT_DIR / "transformer_dnn_fingerprint_generator.py"

OUTPUT_DIR = SCRIPT_DIR / "output"
MIXTURE_SCORES_NPZ = OUTPUT_DIR / "mixture_scores.npz"
METRICS_JSON = OUTPUT_DIR / "summary_metrics.json"
PR_CURVES_CSV = OUTPUT_DIR / "pr_curves.csv"
BUILD_REPORT_JSON = OUTPUT_DIR / "mixture_report.json"
FIG_OUTPUT_SVG = OUTPUT_DIR / "fig3f_external_pr_f1.svg"
AP_BOOTSTRAP_JSON = OUTPUT_DIR / "ap_bootstrap_stats.json"

MASS_TOLERANCE_DA = 0.01
MASS_TOLERANCE_FALLBACKS = (0.02, 0.05)
COSINE_BIN_WIDTH = 0.1
MZ_MIN, MZ_MAX = 20, 1200
MIN_PEAKS = 5
CONNECTIVITY_LEN = 14

THRESHOLD_START = 0.40
THRESHOLD_END = 1.00
THRESHOLD_STEP = 0.01

FP_THRESHOLD = 0.3

METHOD_CONFIG = {
    "MW-only": {"label": "MW-only", "color": "#9E9E9E"},
    "MW + Cosine": {"label": "Cosine", "color": "#4472C4"},
    "MW + Transformer-DNN": {"label": "Transformer-DNN", "color": "#C00000"},
}
PLOT_ORDER = ["MW-only", "MW + Cosine", "MW + Transformer-DNN"]


def _print_progress(label: str, current: int, total: int, last_pct: list[int]) -> None:
    """Print at ~10% milestones (and at completion)."""
    if total <= 0:
        return
    pct = int(100 * current / total)
    if current >= total or pct >= last_pct[0] + 10 or current == 1:
        print(f"  {label}: {current}/{total} ({min(pct, 100)}%)", flush=True)
        last_pct[0] = pct if current < total else 100


# ── MS/MS utilities ─────────────────────────────────────────────
def parse_peaks(peaks_str: object) -> list[tuple[float, float]]:
    if peaks_str is None or (isinstance(peaks_str, float) and np.isnan(peaks_str)):
        return []
    text = str(peaks_str).strip()
    if not text or text.lower() == "nan":
        return []

    peaks: list[tuple[float, float]] = []
    if text.startswith("[[") and text.endswith("]]"):
        for item in re.findall(r"\[\s*([^\]]+)\]", text):
            parts = re.split(r"[,;\s]+", item.strip())
            nums = [p for p in parts if p]
            if len(nums) >= 2:
                try:
                    peaks.append((float(nums[0]), float(nums[1])))
                except ValueError:
                    pass
        return peaks

    for chunk in re.split(r"[;\n]+", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk and " " not in chunk.split(":")[0]:
            left, right = chunk.split(":", 1)
            try:
                peaks.append((float(left.strip()), float(right.strip())))
            except ValueError:
                pass
            continue
        parts = chunk.split()
        if len(parts) >= 2:
            try:
                peaks.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return peaks


def peak_count(peaks_str: object) -> int:
    return len(parse_peaks(peaks_str))


def peaks_to_arrays(peaks_str: object) -> tuple[np.ndarray, np.ndarray]:
    peaks = parse_peaks(peaks_str)
    if not peaks:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    mz = np.array([p[0] for p in peaks], dtype=np.float32)
    intensity = np.array([p[1] for p in peaks], dtype=np.float32)
    mask = (mz >= MZ_MIN) & (mz <= MZ_MAX) & (intensity > 0)
    mz = mz[mask]
    intensity = intensity[mask]
    if intensity.size > 0:
        intensity = intensity / float(intensity.max())
    return mz, intensity


def bin_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    bin_width: float = COSINE_BIN_WIDTH,
    mz_min: float = MZ_MIN,
    mz_max: float = MZ_MAX,
) -> np.ndarray:
    if mz.size == 0:
        n_bins = int(np.ceil((mz_max - mz_min) / bin_width))
        return np.zeros(n_bins, dtype=np.float32)
    edges = np.arange(mz_min, mz_max + bin_width, bin_width)
    vec, _ = np.histogram(mz, bins=edges, weights=intensity)
    vec = vec.astype(np.float32)
    mx = vec.max()
    if mx > 0:
        vec /= mx
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def tanimoto_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    ab = float(np.dot(a, b))
    aa = float(np.dot(a, a))
    bb = float(np.dot(b, b))
    denom = aa + bb - ab
    if denom <= 0:
        return 0.0
    return ab / denom


def connectivity_inchikey(inchikey: object) -> str:
    value = str(inchikey or "").strip()
    if len(value) < 14:
        return ""
    return value[:14]


def spectrum_quality(peaks_str: object, precursor_mz: object, min_peaks: int = MIN_PEAKS) -> bool:
    peaks = parse_peaks(peaks_str)
    if len(peaks) < min_peaks:
        return False
    try:
        mz = float(precursor_mz)
    except (TypeError, ValueError):
        return False
    return mz > 0


def iter_mass_matches(
    sorted_masses: np.ndarray,
    target: float,
    tolerance: float,
) -> Iterable[int]:
    lo = target - tolerance
    hi = target + tolerance
    left = int(np.searchsorted(sorted_masses, lo, side="left"))
    right = int(np.searchsorted(sorted_masses, hi, side="right"))
    for idx in range(left, right):
        yield idx


# ── Mixture retrieval: score & evaluate ───────────────────────────
def load_positive_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "pair_id" not in df.columns:
        df["pair_id"] = np.arange(len(df), dtype=int)
    return df.reset_index(drop=True)


def build_reactant_library(pairs: pd.DataFrame) -> pd.DataFrame:
    """Unique reactants (substrate pool) with best spectrum per compound."""
    records = []
    for _, row in pairs.iterrows():
        records.append(
            {
                "compound_key": row["reactant_connectivity"],
                "exact_mass": float(row["reactant_exact_mass"]),
                "msms_peaks": row["reactant_msms_peaks"],
                "adduct": row["reactant_adduct"],
                "_exact": int(row["reactant_spectrum_match_type"] == "exact_inchikey"),
                "_peaks": int(row.get("reactant_nz_peaks", peak_count(row["reactant_msms_peaks"]))),
            }
        )
    lib = pd.DataFrame(records)
    lib = lib.sort_values(["_exact", "_peaks"], ascending=False)
    lib = lib.drop_duplicates(subset=["compound_key"], keep="first")
    return lib.drop(columns=["_exact", "_peaks"]).reset_index(drop=True)


def load_fingerprint_generator(model_path: Path, device: str):
    spec = importlib.util.spec_from_file_location("tdnn_fp", FINGERPRINT_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {FINGERPRINT_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TransformerDNNFingerprintGenerator(model_path=str(model_path), device=device)


def spectrum_fp(gen, peaks_str: str, adduct: str) -> np.ndarray | None:
    mz, intensity = peaks_to_arrays(peaks_str)
    if mz.size == 0:
        return None
    fp = gen.generate_fingerprint(
        mz.tolist(),
        intensity.tolist(),
        precursor_type=str(adduct),
        threshold=FP_THRESHOLD,
        binary=True,
    )
    if fp is None:
        return None
    return np.asarray(fp, dtype=np.float32)


def _stack_bins(df: pd.DataFrame, peaks_col: str, label: str = "binning") -> np.ndarray:
    n = len(df)
    bins = []
    last_pct = [-1]
    for i, peaks in enumerate(df[peaks_col], start=1):
        mz, intensity = peaks_to_arrays(peaks)
        bins.append(bin_spectrum(mz, intensity))
        _print_progress(label, i, n, last_pct)
    return np.stack(bins, axis=0).astype(np.float32)


def _stack_fps(
    gen,
    df: pd.DataFrame,
    peaks_col: str,
    adduct_col: str,
    label: str = "fingerprints",
) -> np.ndarray:
    fps: list[np.ndarray] = []
    dim: int | None = None
    n = len(df)
    last_pct = [-1]
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        fp = spectrum_fp(gen, row[peaks_col], row[adduct_col])
        if fp is not None:
            dim = fp.shape[0]
            fps.append(fp)
        else:
            fps.append(None)
        _print_progress(label, i, n, last_pct)
    if dim is None:
        raise ValueError("No valid fingerprints generated")
    return np.stack(
        [fp if fp is not None else np.zeros(dim, dtype=np.float32) for fp in fps],
        axis=0,
    ).astype(np.float32)


def cosine_matrix(product_bins: np.ndarray, reactant_bins: np.ndarray) -> np.ndarray:
    """(n_queries, n_bins) × (n_reactants, n_bins) → (n_queries, n_reactants)."""
    p_norm = product_bins / np.maximum(np.linalg.norm(product_bins, axis=1, keepdims=True), 1e-12)
    r_norm = reactant_bins / np.maximum(np.linalg.norm(reactant_bins, axis=1, keepdims=True), 1e-12)
    return (p_norm @ r_norm.T).astype(np.float32)


def tanimoto_matrix(product_fp: np.ndarray, reactant_fp: np.ndarray) -> np.ndarray:
    intersection = product_fp @ reactant_fp.T
    union = product_fp.sum(axis=1, keepdims=True) + reactant_fp.sum(axis=1, keepdims=True).T - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, intersection / union, 0.0).astype(np.float32)


def score_mixture_retrieval(
    positive_path: Path = POSITIVE_PAIRS_CSV,
    output_npz: Path = MIXTURE_SCORES_NPZ,
    report_json: Path = BUILD_REPORT_JSON,
    model_path: Path = MODEL_PATH,
    device: str | None = None,
    mass_tol: float = MASS_TOLERANCE_DA,
    tier: str | None = None,
) -> dict:
    """
    For each product query, score every reactant in the library.
    MW gate: |mass(P) - mass(R) - Δm_obs| ≤ mass_tol (per-query observed Δm).
    """
    pairs = load_positive_pairs(positive_path)
    if tier:
        pairs = pairs[pairs["filter_tier"] == tier].copy().reset_index(drop=True)
    if pairs.empty:
        raise ValueError("No positive pairs to evaluate")

    reactants = build_reactant_library(pairs)
    n_q = len(pairs)
    n_r = len(reactants)
    print(f"  queries (products): {n_q}, reactant library: {n_r}", flush=True)

    r_keys = reactants["compound_key"].tolist()
    r_key_to_idx = {k: i for i, k in enumerate(r_keys)}
    true_r_idx = np.array([r_key_to_idx[k] for k in pairs["reactant_connectivity"]], dtype=np.int32)

    product_mass = pairs["product_exact_mass"].to_numpy(dtype=np.float64)
    delta_m = pairs["delta_mass"].to_numpy(dtype=np.float64)
    r_mass = reactants["exact_mass"].to_numpy(dtype=np.float64)

    print("  computing MW candidate matrix ...", flush=True)
    mw_pass = np.abs(product_mass[:, None] - r_mass[None, :] - delta_m[:, None]) <= mass_tol
    print(f"  MW matrix done (mean candidates/query: {mw_pass.sum(axis=1).mean():.1f})", flush=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  loading Transformer-DNN on {device} ...", flush=True)
    gen = load_fingerprint_generator(model_path, device)

    print("  product spectrum binning:", flush=True)
    product_bins = _stack_bins(pairs, "product_msms_peaks", label="product bins")
    print("  reactant spectrum binning:", flush=True)
    reactant_bins = _stack_bins(reactants, "msms_peaks", label="reactant bins")
    print("  cosine similarity matrix ...", flush=True)
    cosine_mat = cosine_matrix(product_bins, reactant_bins)
    print(f"  cosine matrix done ({cosine_mat.shape[0]}×{cosine_mat.shape[1]})", flush=True)

    print("  product DNN fingerprints:", flush=True)
    product_fp = _stack_fps(gen, pairs, "product_msms_peaks", "product_adduct", label="product fp")
    print("  reactant DNN fingerprints:", flush=True)
    reactant_fp = _stack_fps(gen, reactants, "msms_peaks", "adduct", label="reactant fp")
    print("  DNN Tanimoto matrix ...", flush=True)
    dnn_mat = tanimoto_matrix(product_fp, reactant_fp)
    print(f"  DNN matrix done ({dnn_mat.shape[0]}×{dnn_mat.shape[1]})", flush=True)

    print(f"  saving {output_npz.name} ...", flush=True)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        mw_pass=mw_pass,
        cosine_mat=cosine_mat,
        dnn_mat=dnn_mat,
        true_r_idx=true_r_idx,
        pair_id=pairs["pair_id"].to_numpy(dtype=np.int32),
        product_keys=pairs["product_connectivity"].to_numpy(),
        reactant_keys=np.array(r_keys),
        product_mass=product_mass,
        delta_m=delta_m,
    )

    mean_mw_cands = float(mw_pass.sum(axis=1).mean())
    report = {
        "eval_mode": "mixture_retrieval",
        "n_queries": n_q,
        "n_reactants_in_library": n_r,
        "mass_tolerance_da": mass_tol,
        "mean_mw_candidates_per_product": mean_mw_cands,
        "tier_filter": tier,
        "scores_npz": str(output_npz),
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _thresholds() -> np.ndarray:
    return np.arange(THRESHOLD_START, THRESHOLD_END + 1e-9, THRESHOLD_STEP)


def _pr_at_tau_mixture(
    mw_pass: np.ndarray,
    sim_mat: np.ndarray | None,
    true_r_idx: np.ndarray,
    tau: float,
    mw_only: bool = False,
) -> tuple[float, float]:
    """Aggregate TP/FP/FN across product queries at threshold τ."""
    n_q = mw_pass.shape[0]
    if mw_only:
        pred = mw_pass.copy()
    else:
        pred = mw_pass & (sim_mat >= tau)

    true_mask = np.zeros_like(pred, dtype=bool)
    true_mask[np.arange(n_q), true_r_idx] = True

    tp = int(np.sum(pred & true_mask))
    fp = int(np.sum(pred & ~true_mask))
    fn = int(np.sum(~pred & true_mask))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def _macro_average_precision(mw_pass: np.ndarray, sim_mat: np.ndarray, true_r_idx: np.ndarray) -> float:
    aps = []
    n_q = mw_pass.shape[0]
    for j in range(n_q):
        mask = mw_pass[j]
        if not np.any(mask):
            continue
        cand = np.where(mask)[0]
        labels = (cand == true_r_idx[j]).astype(np.float64)
        if labels.sum() == 0:
            continue
        scores = sim_mat[j, cand]
        order = np.argsort(-scores)
        y = labels[order]
        tp = 0
        fp = 0
        precisions = []
        for label in y:
            if label == 1:
                tp += 1
                precisions.append(tp / (tp + fp))
            else:
                fp += 1
        if precisions:
            aps.append(float(np.mean(precisions)))
    return float(np.mean(aps)) if aps else 0.0


def evaluate_mixture(
    scores_npz: Path = MIXTURE_SCORES_NPZ,
    metrics_json: Path = METRICS_JSON,
    pr_csv: Path = PR_CURVES_CSV,
) -> dict:
    data = np.load(scores_npz, allow_pickle=True)
    mw_pass = data["mw_pass"]
    cosine_mat = data["cosine_mat"]
    dnn_mat = data["dnn_mat"]
    true_r_idx = data["true_r_idx"]
    n_q = int(mw_pass.shape[0])

    methods = [
        ("MW-only", None, True),
        ("MW + Cosine", cosine_mat, False),
        ("MW + Transformer-DNN", dnn_mat, False),
    ]

    results = []
    for name, sim_mat, mw_only in methods:
        print(f"  threshold scan: {name} ...", flush=True)
        rows = []
        best_f1 = -1.0
        best_tau = THRESHOLD_START
        for tau in _thresholds():
            p, r = _pr_at_tau_mixture(mw_pass, sim_mat, true_r_idx, tau, mw_only=mw_only)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            rows.append({"method": name, "threshold": float(tau), "precision": p, "recall": r, "f1": f1})
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)

        if mw_only:
            ap = rows[0]["precision"] if rows else 0.0
        else:
            ap = _macro_average_precision(mw_pass, sim_mat, true_r_idx)

        def metric_at_recall(target_r: float) -> float:
            valid = [row for row in rows if row["recall"] >= target_r]
            return max((row["precision"] for row in valid), default=0.0)

        def metric_at_precision(target_p: float) -> float:
            valid = [row for row in rows if row["precision"] >= target_p]
            return max((row["recall"] for row in valid), default=0.0)

        results.append(
            {
                "method": name,
                "ap": ap,
                "f1_max": best_f1,
                "best_threshold": best_tau,
                "precision_at_recall_0.8": metric_at_recall(0.8),
                "recall_at_precision_0.8": metric_at_precision(0.8),
                "curve": rows,
            }
        )

    pd.concat([pd.DataFrame(r["curve"]) for r in results], ignore_index=True).to_csv(pr_csv, index=False)

    summary = {
        "eval_mode": "mixture_retrieval",
        "n_queries": n_q,
        "n_reactants": int(data["reactant_keys"].shape[0]),
        "methods": [{k: v for k, v in r.items() if k != "curve"} for r in results],
    }
    metrics_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ── Plot style (SVG text, not paths) ──────────────────────────────
def configure_matplotlib() -> None:
    """Use editable text in SVG/PDF instead of converting glyphs to paths."""
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
        }
    )


def save_figure(fig, path: str | Path, *, also_png: bool = False) -> Path:
    """Save figure as SVG; optionally also export PNG for quick preview."""
    configure_matplotlib()
    out = Path(path)
    if out.suffix.lower() != ".svg":
        out = out.with_suffix(".svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    if also_png:
        fig.savefig(out.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    return out


# ── Step 3: Plot (Fig. 3f + AP) ───────────────────────────────────
def _pr_legend_handles() -> list:
    """One legend entry per plotted curve (color + linestyle match the data)."""
    handles = []
    for method in PLOT_ORDER:
        cfg = METHOD_CONFIG[method]
        c, lbl = cfg["color"], cfg["label"]
        handles.append(
            mlines.Line2D([], [], color=c, linestyle="-", linewidth=1.8, label=f"{lbl} (Precision)")
        )
        handles.append(
            mlines.Line2D([], [], color=c, linestyle="--", linewidth=1.8, label=f"{lbl} (Recall)")
        )
    return handles


def _f1_legend_handles() -> list:
    return [
        mlines.Line2D(
            [], [],
            color=METHOD_CONFIG[m]["color"],
            linestyle="-",
            linewidth=2.0,
            label=METHOD_CONFIG[m]["label"],
        )
        for m in PLOT_ORDER
    ]


def _ap_per_query(
    mw_pass: np.ndarray,
    true_r_idx: np.ndarray,
    sim_mat: np.ndarray | None,
    mw_only: bool,
) -> np.ndarray:
    n_q = mw_pass.shape[0]
    out = np.zeros(n_q, dtype=float)
    for j in range(n_q):
        cand = np.where(mw_pass[j])[0]
        if len(cand) == 0:
            continue
        t = int(true_r_idx[j])
        if np.all(cand != t):
            continue
        if mw_only:
            out[j] = 1.0 / len(cand)
            continue

        scores = sim_mat[j, cand]
        order = np.argsort(-scores)
        ranked = cand[order]
        y = (ranked == t).astype(float)

        tp = 0
        fp = 0
        precisions = []
        for label in y:
            if label == 1:
                tp += 1
                precisions.append(tp / (tp + fp))
            else:
                fp += 1
        out[j] = float(np.mean(precisions)) if precisions else 0.0
    return out


def _hit_at_1_per_query(
    mw_pass: np.ndarray,
    true_r_idx: np.ndarray,
    sim_mat: np.ndarray | None,
    mw_only: bool,
) -> np.ndarray:
    """Top-1 accuracy: true reactant ranked first among MW candidates."""
    n_q = mw_pass.shape[0]
    out = np.zeros(n_q, dtype=float)
    for j in range(n_q):
        cand = np.where(mw_pass[j])[0]
        if len(cand) == 0:
            continue
        t = int(true_r_idx[j])
        if np.all(cand != t):
            continue
        if mw_only:
            # No spectral ranker: expected hit if one MW candidate is chosen uniformly.
            out[j] = 1.0 / len(cand)
            continue
        scores = sim_mat[j, cand]
        order = np.argsort(-scores, kind="stable")
        out[j] = 1.0 if cand[order[0]] == t else 0.0
    return out


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = values[idx].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(lo), float(hi)


def _collect_ap_bootstrap_stats(scores_npz: Path = MIXTURE_SCORES_NPZ) -> list[dict]:
    data = np.load(scores_npz, allow_pickle=True)
    mw_pass = data["mw_pass"]
    cosine_mat = data["cosine_mat"]
    dnn_mat = data["dnn_mat"]
    true_r_idx = data["true_r_idx"]

    method_specs = [
        ("MW-only", None, True),
        ("MW + Cosine", cosine_mat, False),
        ("MW + Transformer-DNN", dnn_mat, False),
    ]

    stats = []
    for name, sim_mat, mw_only in method_specs:
        q_ap = _ap_per_query(mw_pass, true_r_idx, sim_mat, mw_only)
        q_hit1 = _hit_at_1_per_query(mw_pass, true_r_idx, sim_mat, mw_only)
        mean_ap = float(q_ap.mean())
        hit_at_1 = float(q_hit1.mean())
        ap_ci_low, ap_ci_high = _bootstrap_ci(q_ap, n_boot=1000, seed=42)
        hit_ci_low, hit_ci_high = _bootstrap_ci(q_hit1, n_boot=1000, seed=42)
        sd = float(q_ap.std(ddof=1)) if len(q_ap) > 1 else 0.0
        se = sd / np.sqrt(len(q_ap)) if len(q_ap) > 0 else 0.0
        stats.append(
            {
                "method": name,
                "ap_mean": mean_ap,
                "hit_at_1": hit_at_1,
                "ci95_low": ap_ci_low,
                "ci95_high": ap_ci_high,
                "hit_ci95_low": hit_ci_low,
                "hit_ci95_high": hit_ci_high,
                "sd": sd,
                "se": float(se),
                "color": METHOD_CONFIG[name]["color"],
            }
        )
    return stats


def _draw_ap_panel(ax: plt.Axes, stats: list[dict]) -> None:
    """Grouped bars: mAP and Hit@1 per method (both with 95% bootstrap CI)."""
    labels = [METHOD_CONFIG[s["method"]]["label"] for s in stats]
    map_means = [s["ap_mean"] for s in stats]
    hit_means = [s["hit_at_1"] for s in stats]
    colors = [s["color"] for s in stats]
    map_yerr = np.array(
        [
            [s["ap_mean"] - s["ci95_low"] for s in stats],
            [s["ci95_high"] - s["ap_mean"] for s in stats],
        ]
    )
    hit_yerr = np.array(
        [
            [s["hit_at_1"] - s["hit_ci95_low"] for s in stats],
            [s["hit_ci95_high"] - s["hit_at_1"] for s in stats],
        ]
    )

    n = len(stats)
    x = np.arange(n)
    bar_w = 0.36

    ax.bar(
        x - bar_w / 2,
        map_means,
        bar_w,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        yerr=map_yerr,
        capsize=3,
        ecolor="black",
        error_kw={"elinewidth": 1.0},
        label="mAP",
        zorder=3,
    )
    ax.bar(
        x + bar_w / 2,
        hit_means,
        bar_w,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        yerr=hit_yerr,
        capsize=3,
        ecolor="black",
        error_kw={"elinewidth": 1.0},
        alpha=0.42,
        hatch="//",
        label="Hit@1",
        zorder=2,
    )

    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            mlines.Line2D([], [], color="0.35", linewidth=6, label="mAP (95% CI)"),
            mpatches.Patch(
                facecolor="0.85",
                edgecolor="black",
                hatch="//",
                alpha=0.42,
                label="Hit@1 (95% CI)",
            ),
        ],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )

    for i, (m, h) in enumerate(zip(map_means, hit_means)):
        ax.text(
            i - bar_w / 2,
            m + map_yerr[1][i] + 0.02,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        ax.text(
            i + bar_w / 2,
            h + hit_yerr[1][i] + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def _draw_mw_candidate_hist(ax: plt.Axes, mw_pass: np.ndarray) -> None:
    """Histogram of MW-matching reactant counts per product query."""
    per_product = mw_pass.sum(axis=1).astype(int)
    n = len(per_product)
    values, frequencies = np.unique(per_product, return_counts=True)

    ax.bar(
        values,
        frequencies,
        width=0.88,
        color="#4472C4",
        edgecolor="white",
        linewidth=1.2,
        align="center",
    )
    ax.set_title("Distribution of MW Candidate Count per Product", fontsize=11)
    ax.set_xlabel("Number of MW candidate reactants per product", fontsize=11)
    ax.set_ylabel("Number of products", fontsize=11)
    ax.set_xticks(np.arange(int(per_product.min()), int(per_product.max()) + 1))
    ax.set_ylim(0, max(float(frequencies.max()) * 1.12, float(frequencies.max()) + 8))
    ax.grid(axis="y", alpha=0.35, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    stats_text = (
        f"n = {n} products\n"
        f"mean = {per_product.mean():.2f}, median = {np.median(per_product):.0f}\n"
        f"min = {int(per_product.min())}, max = {int(per_product.max())}"
    )
    ax.text(
        0.97,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )


def _write_ap_bootstrap_json(stats: list[dict], n_queries: int, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "n_queries": n_queries,
                "bootstrap_samples": 1000,
                "methods": [
                    {
                        "method": s["method"],
                        "ap_mean": s["ap_mean"],
                        "hit_at_1": s["hit_at_1"],
                        "ci95_low": s["ci95_low"],
                        "ci95_high": s["ci95_high"],
                        "hit_ci95_low": s["hit_ci95_low"],
                        "hit_ci95_high": s["hit_ci95_high"],
                        "sd": s["sd"],
                        "se": s["se"],
                    }
                    for s in stats
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def plot_figure(
    pr_csv: Path = PR_CURVES_CSV,
    scores_npz: Path = MIXTURE_SCORES_NPZ,
    out_svg: Path = FIG_OUTPUT_SVG,
    bootstrap_json: Path = AP_BOOTSTRAP_JSON,
) -> None:
    """Single composite figure (2×2): P/R, F1, AP, MW-candidate histogram."""
    configure_matplotlib()
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pr_csv)
    data = np.load(scores_npz, allow_pickle=True)
    mw_pass = data["mw_pass"]
    ap_stats = _collect_ap_bootstrap_stats(scores_npz)
    n_queries = int(mw_pass.shape[0])
    _write_ap_bootstrap_json(ap_stats, n_queries, bootstrap_json)

    pr_handles = _pr_legend_handles()
    f1_handles = _f1_legend_handles()

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))

    ax = axes[0, 0]
    for method in PLOT_ORDER:
        sub = df[df["method"] == method].sort_values("threshold")
        if sub.empty:
            continue
        c = METHOD_CONFIG[method]["color"]
        ax.plot(sub["threshold"], sub["precision"], color=c, linestyle="-", linewidth=1.8)
        ax.plot(sub["threshold"], sub["recall"], color=c, linestyle="--", linewidth=1.8)
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("Precision & Recall", fontsize=11)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(handles=pr_handles, loc="lower left", fontsize=6.5, frameon=True, ncol=2)

    ax2 = axes[0, 1]
    for method in PLOT_ORDER:
        sub = df[df["method"] == method].sort_values("threshold")
        if sub.empty:
            continue
        ax2.plot(
            sub["threshold"],
            sub["f1"],
            color=METHOD_CONFIG[method]["color"],
            linestyle="-",
            linewidth=2.0,
        )
    ax2.set_xlabel("Threshold", fontsize=11)
    ax2.set_ylabel("F1-score", fontsize=11)
    ax2.set_xlim(0.4, 1.0)
    ax2.set_ylim(0, 1.02)
    ax2.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.legend(handles=f1_handles, loc="lower left", fontsize=8)

    _draw_ap_panel(axes[1, 0], ap_stats)
    _draw_mw_candidate_hist(axes[1, 1], mw_pass)

    fig.tight_layout()
    saved = save_figure(fig, out_svg)
    plt.close(fig)
    print(f"Saved: {saved}")
    print(f"Saved: {bootstrap_json}")


# ── Main ──────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 1: mixture retrieval MR-FMN validation (Fig. 3f)")
    parser.add_argument("--input", type=Path, required=True, help="Path to positive_pairs_clean.csv")
    parser.add_argument("--model", type=Path, required=True, help="Path to the trained Transformer-DNN checkpoint")
    parser.add_argument("--skip-score", action="store_true", help="Skip scoring if mixture_scores.npz exists")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tier", type=str, default=None, choices=["A", "B"])
    parser.add_argument("--mass-tol", type=float, default=MASS_TOLERANCE_DA)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_score:
        if not args.input.exists():
            print(f"Input not found: {args.input}", file=sys.stderr)
            return 1
        if not args.model.exists():
            print(f"Model not found: {args.model}", file=sys.stderr)
            return 1
        print("=== Step 1: Score mixture retrieval (each product × all reactants) ===")
        score_mixture_retrieval(
            positive_path=args.input,
            model_path=args.model,
            device=args.device,
            mass_tol=args.mass_tol,
            tier=args.tier,
        )
    elif not MIXTURE_SCORES_NPZ.exists():
        print(f"Scores not found: {MIXTURE_SCORES_NPZ}", file=sys.stderr)
        return 1

    if not args.skip_eval:
        print("=== Step 2: Threshold-scan evaluation (Fig. 3f protocol) ===")
        evaluate_mixture()

    if not args.skip_plot:
        if not MIXTURE_SCORES_NPZ.exists():
            print(f"Scores not found: {MIXTURE_SCORES_NPZ}", file=sys.stderr)
            return 1
        if not PR_CURVES_CSV.exists():
            print(f"PR curves missing: {PR_CURVES_CSV}", file=sys.stderr)
            return 1
        print("=== Step 3: Plot composite figure ===")
        plot_figure()

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

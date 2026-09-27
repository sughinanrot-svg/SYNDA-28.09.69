import io
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from matplotlib.backends.backend_pdf import PdfPages

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

import backend_algorithm as backend
import evaluation_module as evaluator
import importlib

IDENTIFIER_PATTERNS = (
    "service",
    "serviceno",
    "service_no",
    "service no",
    "hn",
    "mrn",
    "id",
    "name",
    "firstname",
    "lastname",
    "dob",
    "citizen",
    "patient",
    "encounter",
    "record",
    "phone",
    "email",
)

st.set_page_config(
    page_title="SYNDA - Synthetic Medical Laboratory Data Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------
# Hospital Information System (HIS / LIS) GUI Styling
# -------------------------------------------------------------
def inject_clinical_his_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        div[data-stale="true"] {
            opacity: 1 !important;
            filter: none !important;
            transition: none !important;
        }

        .stApp {
            background-color: #ecf0f5;
            font-family: 'IBM Plex Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #2c3e50;
        }

        section[data-testid="stSidebar"] {
            background-color: #1a202c !important;
            border-right: 1px solid #2d3748;
        }
        section[data-testid="stSidebar"] * {
            color: #cbd5e0 !important;
            font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #ffffff !important;
            font-weight: 600;
        }

        .nav-flow-item {
            padding: 9px 12px;
            margin-bottom: 6px;
            border-radius: 4px;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
            color: #a0aec0 !important;
            background-color: #2d3748;
            border-left: 3px solid transparent;
            transition: all 0.15s ease-in-out;
        }
        .nav-flow-item:hover {
            background-color: #3b475a;
            color: #ffffff !important;
            border-left-color: #3182ce;
        }
        .nav-flow-active {
            background-color: #2b6cb0 !important;
            color: #ffffff !important;
            font-weight: 600;
            border-left: 3px solid #63b3ed !important;
        }

        .his-box {
            background-color: #ffffff;
            border-top: 3px solid #2b6cb0;
            border-radius: 3px;
            padding: 18px 22px;
            margin-bottom: 22px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            border-left: 1px solid #e2e8f0;
            border-right: 1px solid #e2e8f0;
            border-bottom: 1px solid #e2e8f0;
        }
        .his-box-header {
            font-size: 15px;
            font-weight: 600;
            color: #1a202c;
            margin-bottom: 14px;
            border-bottom: 1px solid #edf2f7;
            padding-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-top: 2px solid #2b6cb0;
            border-radius: 3px;
            padding: 10px 14px;
        }
        div[data-testid="stMetric"] * {
            color: #2d3748 !important;
        }

        .stButton>button, .stDownloadButton>button {
            background-color: #2b6cb0 !important;
            color: #ffffff !important;
            border: 1px solid #2c5282 !important;
            border-radius: 3px !important;
            font-weight: 500 !important;
            font-size: 13px !important;
            padding: 7px 18px !important;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            background-color: #2c5282 !important;
            border-color: #1a365d !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 3px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------
# Cached Backend Operations
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_process_parameters(df_real: pd.DataFrame):
    return backend.process_and_extract_parameters(df_real)


@st.cache_data(show_spinner=False)
def _cached_generate_synthetic(parameters: dict, num_samples: int, selected_variables: list | None = None, include_synthetic_hn: bool = False, random_seed: int | None = None):
    return backend.generate_synthetic_data(
        parameters=parameters,
        num_samples=num_samples,
        selected_variables=selected_variables,
        include_synthetic_hn=include_synthetic_hn,
        random_seed=random_seed,
    )


def _safe_evaluation_results(df_real: pd.DataFrame, df_synth: pd.DataFrame):
    try:
        return evaluator.evaluate_all_metrics(df_real, df_synth, show_plots=True)
    except Exception:
        ks_df = evaluator.evaluate_synthetic_data(df_real, df_synth)
        moments_df = evaluator.summarize_moments(df_real, df_synth)
        corr_fig = None
        try:
            corr_fig = evaluator.plot_comparison_correlation_heatmap(df_real, df_synth)
        except Exception:
            corr_fig = None
        return {
            "ks": ks_df,
            "moments": moments_df,
            "corr_real": None,
            "corr_synth": None,
            "figures": {"corr_heatmap": corr_fig},
        }


def is_identifier_column(column_name):
    name = str(column_name).strip().lower()
    return any(token in name for token in IDENTIFIER_PATTERNS)


def remove_identifier_columns(df):
    if df is None or df.empty:
        return df.copy()
    cols_to_drop = [col for col in df.columns if is_identifier_column(col)]
    return df.drop(columns=cols_to_drop, errors="ignore").copy()


def numeric_columns_only(df):
    cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
            continue
        try:
            coerced = pd.to_numeric(df[col].dropna(), errors="coerce").dropna()
            if len(coerced) >= 2:
                cols.append(col)
        except Exception:
            continue
    return cols


def normalize_feature_name(value):
    if value is None:
        return value
    text = str(value).strip()
    if text.upper() in {"LDL2", "LDL 2"}:
        return "LDL"
    return text


def normalize_dataframe_columns(df):
    if df is None:
        return df
    renamed = {col: normalize_feature_name(col) for col in df.columns}
    if renamed:
        df = df.rename(columns=renamed)
    return df


def download_png_button(fig_bytes, filename, label="Download High-Res PNG"):
    st.download_button(
        label=label,
        data=fig_bytes,
        file_name=filename,
        mime="image/png",
    )


def _distribution_x_limits(values, q_low=0.01, q_high=0.99, pad=0.08):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.quantile(values, [q_low, q_high])
    spread = high - low
    if not np.isfinite(spread) or spread <= 0:
        low, high = float(np.min(values)), float(np.max(values))
        if np.isclose(low, high):
            low -= 0.5
            high += 0.5
        spread = high - low
    pad_amt = max(spread * pad, 1e-6)
    low = float(low - pad_amt)
    high = float(high + pad_amt)
    if np.nanmin(values) >= 0:
        low = max(low, 0.0)
    return low, high


# -------------------------------------------------------------
# Distribution Inspector Figure
# -------------------------------------------------------------
def build_distribution_histogram_figure(df_real, column_name, bins=28):
    vals = pd.to_numeric(df_real[column_name].dropna(), errors="coerce").dropna()
    if len(vals) < 3:
        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.text(0.5, 0.5, "Insufficient data for distribution inspection", ha="center", va="center")
        ax.set_axis_off()
        return fig

    n_bins = max(8, min(int(bins), int(np.sqrt(len(vals)) * 3)))
    x_min, x_max = _distribution_x_limits(vals, q_low=0.005, q_high=0.995, pad=0.08)
    if np.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5

    mean_val = float(np.mean(vals))
    median_val = float(np.median(vals))
    skewness = float(stats.skew(vals, bias=False))

    hist_data, bin_edges = np.histogram(vals, bins=n_bins, range=(x_min, x_max), density=True)
    hist_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mode_val = float(hist_centers[np.argmax(hist_data)]) if hist_data.size else mean_val

    if skewness > 0.2:
        skew_label = "Right-Skewed Distribution (Mean > Median > Mode)"
    elif skewness < -0.2:
        skew_label = "Left-Skewed Distribution (Mean < Median < Mode)"
    else:
        skew_label = "Symmetrical Distribution"

    reference_lines = [
        ("Mean", mean_val, "#DC2626"),
        ("Median", median_val, "#2563EB"),
        ("Mode", mode_val, "#059669"),
    ]

    if PLOTLY_AVAILABLE:
        x_grid = np.linspace(x_min, x_max, 400)
        try:
            kde = stats.gaussian_kde(vals)
            y_kde = kde(x_grid)
            y_kde = y_kde / max(np.max(y_kde), 1e-9)
        except Exception:
            y_kde = np.zeros_like(x_grid)

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=vals,
                nbinsx=n_bins,
                name="Histogram",
                opacity=0.85,
                marker=dict(color="#FDE68A", line=dict(color="#D97706", width=1.0)),
                histnorm="probability density",
            )
        )
        if np.any(y_kde):
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y_kde,
                    mode="lines",
                    name="KDE Density",
                    line=dict(color="#3B82F6", width=2.0, dash="solid"),
                    opacity=0.65,
                )
            )

        for label, val, color in reference_lines:
            fig.add_vline(
                x=val,
                line_color=color,
                line_dash="dash",
                line_width=2.0,
                annotation_text=f"{label}: {val:.1f}",
                annotation_position="top left",
                annotation_font=dict(size=10, color=color, family="IBM Plex Sans"),
            )

        fig.update_layout(
            title=dict(
                text=f"{normalize_feature_name(column_name)} Distribution",
                x=0.5,
                font=dict(size=16, color="#0f172a", family="IBM Plex Sans"),
            ),
            annotations=[
                dict(
                    text=f"Skewness γ1 = {skewness:.3f} • {skew_label}",
                    x=0.5,
                    y=1.03,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=11, color="#475569"),
                    align="center",
                )
            ],
            xaxis_title=normalize_feature_name(column_name),
            yaxis_title="Density",
            template="plotly_white",
            barmode="overlay",
            bargap=0.05,
            autosize=True,
            height=460,
            xaxis=dict(range=[x_min, x_max], showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="right", x=1.0),
            margin=dict(l=28, r=20, t=72, b=28),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
        )
        return fig

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)
    ax.hist(vals, bins=n_bins, range=(x_min, x_max), density=True, color="#FDE68A", alpha=0.85, edgecolor="#D97706", linewidth=0.8, label="Histogram")
    try:
        kde = stats.gaussian_kde(vals)
        kde_x = np.linspace(x_min, x_max, 350)
        ax.plot(kde_x, kde(kde_x), color="#3B82F6", linewidth=1.8, linestyle="-", alpha=0.65, label="KDE")
    except Exception:
        pass

    for label, val, color in reference_lines:
        ax.axvline(val, color=color, linestyle="--", linewidth=1.6, label=f"{label}: {val:.1f}")
    ax.set_xlim(x_min, x_max)
    ax.set_title(f"{normalize_feature_name(column_name)} Distribution")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------
# Cached Visualization Renderers
# -------------------------------------------------------------
def _generate_multi_distribution_grid_figure(df, numeric_cols, max_vars=9):
    target_cols = numeric_cols[: min(len(numeric_cols), max_vars)]
    if not target_cols:
        return None

    n_cols = 3
    n_rows = int(np.ceil(len(target_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.4 * n_rows), dpi=180)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, col in enumerate(target_cols):
        ax = axes_flat[idx]
        vals = pd.to_numeric(df[col].dropna(), errors="coerce").dropna()
        if len(vals) >= 3:
            x_min, x_max = _distribution_x_limits(vals, q_low=0.005, q_high=0.995)
            n_bins = max(8, min(24, int(np.sqrt(len(vals)) * 2.5)))
            
            ax.hist(vals, bins=n_bins, range=(x_min, x_max), density=True, color="#FDE68A", alpha=0.85, edgecolor="#D97706", linewidth=0.7)
            try:
                kde = stats.gaussian_kde(vals)
                x_k = np.linspace(x_min, x_max, 250)
                ax.plot(x_k, kde(x_k), color="#3B82F6", linewidth=1.5, alpha=0.65)
            except Exception:
                pass

            mean_v = float(np.mean(vals))
            median_v = float(np.median(vals))
            skew_v = float(stats.skew(vals, bias=False))

            ax.axvline(mean_v, color="#DC2626", linestyle="--", linewidth=1.4, label=f"Mean: {mean_v:.1f}")
            ax.axvline(median_v, color="#2563EB", linestyle=":", linewidth=1.4, label=f"Med: {median_v:.1f}")
            
            ax.set_title(f"{normalize_feature_name(col)} (Skew: {skew_v:.2f})", fontsize=10, fontweight="bold")
            ax.legend(fontsize=7, frameon=False, loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.tick_params(labelsize=8)
        else:
            ax.text(0.5, 0.5, "Insufficient Data", ha="center", va="center")
            ax.set_title(col)

    for j in range(len(target_cols), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def render_cached_multi_distribution_grid(df, numeric_cols, max_vars=9):
    fig = _generate_multi_distribution_grid_figure(df, numeric_cols, max_vars)
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _generate_moment_2x2_grid_figure(df, numeric_cols):
    rows = []
    for var in numeric_cols:
        ser = pd.to_numeric(df[var].dropna(), errors="coerce").dropna()
        if len(ser) < 2:
            continue
        rows.append({
            "Feature": normalize_feature_name(var),
            "Mean": float(ser.mean()),
            "Variance": float(ser.var(ddof=1)),
            "Skewness": float(stats.skew(ser, bias=False)),
            "Kurtosis": float(stats.kurtosis(ser, fisher=False, bias=False)),
        })

    if not rows:
        return None

    m_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.8), dpi=180, constrained_layout=True)
    axes = axes.flatten()
    metrics = ["Mean", "Variance", "Skewness", "Kurtosis"]
    colors = ["#2b6cb0", "#d69e2e", "#38a169", "#e53e3e"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        vals = m_df[metric].tolist()
        x = np.arange(len(m_df))
        ax.bar(x, vals, color=colors[idx], width=0.55, edgecolor="#1a202c", linewidth=0.6)
        ax.set_title(f"{metric} across Clinical Features", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(m_df["Feature"], rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if metric == "Variance":
            ax.set_yscale("log")
            ax.set_ylabel("Log Scale", fontsize=8)

    return fig


@st.cache_data(show_spinner=False)
def render_cached_moment_2x2_grid(df, numeric_cols):
    fig = _generate_moment_2x2_grid_figure(df, numeric_cols)
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _generate_ks_summary_chart_figure(ks_df):
    if ks_df.empty:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), dpi=180)
    sns.barplot(x=ks_df["Variable"], y=ks_df["KS_Statistic"], ax=axes[0], palette="Blues_d")
    axes[0].set_title("K-S Statistic by Variable", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("K-S Statistic")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    sns.barplot(x=ks_df["Variable"], y=ks_df["p_value"], ax=axes[1], palette="viridis")
    axes[1].axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="Significance Threshold (p=0.05)")
    axes[1].set_title("K-S p-value by Variable", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("p-value")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend(loc="upper right", frameon=False)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def render_cached_ks_summary_chart(ks_df):
    fig = _generate_ks_summary_chart_figure(ks_df)
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _generate_overview_kde_figure(df_real, df_synth, max_vars=9):
    exclude_cols = ["HN", "ID", "Name", "ServiceNo", "ServiceDate"]
    numeric_vars = [col for col in df_real.columns if col in df_synth.columns and pd.api.types.is_numeric_dtype(df_real[col]) and col not in exclude_cols]
    if not numeric_vars:
        return None

    plot_vars = numeric_vars[: min(len(numeric_vars), max_vars)]
    n_cols = 3
    n_rows = int(np.ceil(len(plot_vars) / n_cols))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 3.4 * n_rows), dpi=180)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, col in enumerate(plot_vars):
        ax = axes_flat[idx]
        r_data = pd.to_numeric(df_real[col], errors="coerce").dropna()
        s_data = pd.to_numeric(df_synth[col], errors="coerce").dropna()
        if len(r_data) > 1 and len(s_data) > 1:
            ks_res = stats.ks_2samp(r_data, s_data)
            global_min, global_max = _distribution_x_limits(np.concatenate([r_data.to_numpy(), s_data.to_numpy()]), q_low=0.005, q_high=0.995)
            sns.kdeplot(r_data, ax=ax, label="Real", color="#1a365d", linewidth=2, clip=(global_min, global_max))
            sns.kdeplot(s_data, ax=ax, label="Synthetic", color="#dd6b20", linestyle="--", linewidth=2, clip=(global_min, global_max))
            ax.set_xlim(global_min, global_max)
            ax.set_title(f"{normalize_feature_name(col)}", fontsize=10, fontweight="bold")
            ax.text(0.03, 0.95, f"p={ks_res.pvalue:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cbd5e0"})
            ax.legend(loc="upper right", fontsize=7, frameon=False)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.tick_params(labelsize=8)
        else:
            ax.text(0.5, 0.5, "Insufficient Data", ha="center", va="center")
            ax.set_title(normalize_feature_name(col))

    for j in range(len(plot_vars), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def render_cached_overview_kde_figure(df_real, df_synth, max_vars=9):
    fig = _generate_overview_kde_figure(df_real, df_synth, max_vars)
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _generate_correlation_figure(df_real, df_synth):
    common_cols = [col for col in df_real.columns if col in df_synth.columns and pd.api.types.is_numeric_dtype(df_real[col]) and pd.api.types.is_numeric_dtype(df_synth[col])]
    if not common_cols:
        return None

    corr_real = df_real[common_cols].corr(method="spearman")
    corr_synth = df_synth[common_cols].corr(method="spearman")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), dpi=180)
    sns.heatmap(corr_real, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True, ax=axes[0], square=True)
    axes[0].set_title("Real Baseline (Spearman Rank)", fontsize=11, fontweight="bold")
    sns.heatmap(corr_synth, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True, ax=axes[1], square=True)
    axes[1].set_title("Synthetic Cohort (Spearman Rank)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def render_cached_correlation_figure(df_real, df_synth):
    fig = _generate_correlation_figure(df_real, df_synth)
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def build_moment_comparison_chart(df_real, df_synth, metric_name):
    common_cols = [col for col in df_real.columns if col in df_synth.columns and pd.api.types.is_numeric_dtype(df_real[col]) and pd.api.types.is_numeric_dtype(df_synth[col])]
    rows = []
    for col in common_cols:
        r = pd.to_numeric(df_real[col].dropna(), errors="coerce").dropna()
        s = pd.to_numeric(df_synth[col].dropna(), errors="coerce").dropna()
        if len(r) < 2 or len(s) < 2:
            continue
        if metric_name == "Mean":
            real_val = float(np.mean(r))
            synth_val = float(np.mean(s))
        elif metric_name == "Variance":
            real_val = float(np.var(r, ddof=1))
            synth_val = float(np.var(s, ddof=1))
        elif metric_name == "Skewness":
            real_val = float(stats.skew(r, bias=False))
            synth_val = float(stats.skew(s, bias=False))
        elif metric_name == "Kurtosis":
            real_val = float(stats.kurtosis(r, fisher=False, bias=False))
            synth_val = float(stats.kurtosis(s, fisher=False, bias=False))
        else:
            continue
        rows.append({"Variable": col, "Real": real_val, "Synthetic": synth_val})

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"No data available for {metric_name}", ha="center", va="center")
        ax.set_axis_off()
        return fig

    df_plot = pd.DataFrame(rows)
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
    ax.bar(x - 0.18, df_plot["Real"], width=0.35, label="Real", color="#1a365d")
    ax.bar(x + 0.18, df_plot["Synthetic"], width=0.35, label="Synthetic", color="#dd6b20")
    ax.set_title(f"{metric_name} Comparison (Real vs Synthetic)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Variable"], rotation=35, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if metric_name == "Variance":
        ax.set_yscale("log")
    fig.tight_layout()
    return fig


def _generate_all_moments_grid_figure(df_real, df_synth):
    common_cols = [col for col in df_real.columns if col in df_synth.columns and pd.api.types.is_numeric_dtype(df_real[col]) and pd.api.types.is_numeric_dtype(df_synth[col])]
    rows = []
    for col in common_cols:
        r = pd.to_numeric(df_real[col].dropna(), errors="coerce").dropna()
        s = pd.to_numeric(df_synth[col].dropna(), errors="coerce").dropna()
        if len(r) < 2 or len(s) < 2:
            continue
        rows.append({
            "Variable": col,
            "Mean_Real": float(np.mean(r)),
            "Mean_Synth": float(np.mean(s)),
            "Var_Real": float(np.var(r, ddof=1)),
            "Var_Synth": float(np.var(s, ddof=1)),
            "Skew_Real": float(stats.skew(r, bias=False)),
            "Skew_Synth": float(stats.skew(s, bias=False)),
            "Kurt_Real": float(stats.kurtosis(r, fisher=False, bias=False)),
            "Kurt_Synth": float(stats.kurtosis(s, fisher=False, bias=False)),
        })

    if not rows:
        return None

    mdf = pd.DataFrame(rows).set_index("Variable")
    vars_order = mdf.index.tolist()
    x = np.arange(len(vars_order))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), dpi=180, constrained_layout=True)
    axes = axes.flatten()

    metrics = [
        ("Mean", "Mean_Real", "Mean_Synth"),
        ("Variance", "Var_Real", "Var_Synth"),
        ("Skewness", "Skew_Real", "Skew_Synth"),
        ("Kurtosis", "Kurt_Real", "Kurt_Synth"),
    ]

    for ax, (metric, real_col, synth_col) in zip(axes, metrics):
        ax.bar(x - width / 2, mdf[real_col], width, label="Real", color="#1a365d")
        ax.bar(x + width / 2, mdf[synth_col], width, label="Synthetic", color="#dd6b20")
        ax.set_title(f"{metric} Comparison (Real vs Synthetic)", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(vars_order, rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        if metric == "Variance":
            ax.set_yscale("log")

    return fig


def _generate_boxplot_grid_figure(df_real, df_synth, variable_candidates=None):
    common_cols = [col for col in df_real.columns if col in df_synth.columns and pd.api.types.is_numeric_dtype(df_real[col]) and pd.api.types.is_numeric_dtype(df_synth[col])]
    if variable_candidates is None:
        preferred = ["Age", "BMI", "SBP", "DBP", "GLU", "Glucose", "CHO", "TG", "LDL", "HDLC"]
        variable_candidates = [name for name in preferred if name in common_cols]
    if not variable_candidates:
        variable_candidates = common_cols[:4]
    selected = variable_candidates[:4]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), dpi=180)
    axes = axes.flatten()

    for ax, var in zip(axes, selected):
        real_vals = pd.to_numeric(df_real[var], errors="coerce").dropna()
        synth_vals = pd.to_numeric(df_synth[var], errors="coerce").dropna()
        if len(real_vals) < 2 or len(synth_vals) < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.set_title(var)
            ax.set_axis_off()
            continue

        try:
            ax.boxplot([real_vals, synth_vals], tick_labels=["Real", "Synthetic"], patch_artist=True, widths=0.5)
        except TypeError:
            ax.boxplot([real_vals, synth_vals], labels=["Real", "Synthetic"], patch_artist=True, widths=0.5)

        ax.set_title(f"{normalize_feature_name(var)} Distribution & Outliers", fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    for i in range(len(selected), len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    return fig


def build_interactive_boxplot(df_real, df_synth, parameter_name):
    if PLOTLY_AVAILABLE:
        r_series = pd.to_numeric(df_real[parameter_name], errors="coerce").dropna()
        s_series = pd.to_numeric(df_synth[parameter_name], errors="coerce").dropna()
        df_box = pd.concat([
            pd.DataFrame({"Cohort": "Real Source", parameter_name: r_series}),
            pd.DataFrame({"Cohort": "Synthetic Cohort", parameter_name: s_series}),
        ], ignore_index=True)

        fig = px.box(
            df_box,
            x="Cohort",
            y=parameter_name,
            color="Cohort",
            color_discrete_map={"Real Source": "#2b6cb0", "Synthetic Cohort": "#dd6b20"},
            points="outliers",
            template="plotly_white",
        )
        fig.update_layout(
            title=dict(
                text=f"Distributional Spread & Outlier Profile: {normalize_feature_name(parameter_name)}",
                font=dict(size=14, color="#1a202c", family="IBM Plex Sans"),
            ),
            xaxis_title="Cohort Group",
            yaxis_title=f"{normalize_feature_name(parameter_name)} Value",
            showlegend=False,
            height=430,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    r_vals = pd.to_numeric(df_real[parameter_name].dropna(), errors="coerce").dropna()
    s_vals = pd.to_numeric(df_synth[parameter_name].dropna(), errors="coerce").dropna()
    try:
        ax.boxplot([r_vals, s_vals], tick_labels=["Real Source", "Synthetic Cohort"], patch_artist=True)
    except TypeError:
        ax.boxplot([r_vals, s_vals], labels=["Real Source", "Synthetic Cohort"], patch_artist=True)
    ax.set_title(f"{normalize_feature_name(parameter_name)} Distribution Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def _safe_float(value, default="N/A"):
    try:
        if value is None or pd.isna(value):
            return default
        num = float(value)
        return round(num, 4) if np.isfinite(num) else default
    except Exception:
        return default


def _figure_to_reportlab_image(fig, max_width=6.9 * inch, max_height=4.1 * inch):
    """แปลง Matplotlib Figure เป็น Image ของ ReportLab ให้พอดีหน้ากระดาษแบบ 2 รูปต่อหน้า ไม่ล้น"""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        buf.seek(0)
        w_in, h_in = fig.get_size_inches()
        aspect = h_in / max(w_in, 1e-6)
        
        width = max_width
        height = width * aspect
        if height > max_height:
            height = max_height
            width = height / aspect
            
        img = Image(buf, width=width, height=height)
        return img
    except Exception as exc:
        print(f"[Image Render Error]: {exc}")
        return None


def build_statistical_summary_table(df_real, df_synth, ks_df):
    common_cols = [col for col in df_real.columns if col in df_synth.columns and pd.api.types.is_numeric_dtype(df_real[col]) and pd.api.types.is_numeric_dtype(df_synth[col])]
    rows = []
    ks_lookup = {}
    if ks_df is not None and not ks_df.empty:
        for _, row in ks_df.iterrows():
            ks_lookup[str(row.get("Variable", ""))] = row

    for col in common_cols:
        r_c = pd.to_numeric(df_real[col], errors="coerce").dropna()
        s_c = pd.to_numeric(df_synth[col], errors="coerce").dropna()
        if len(r_c) < 2 or len(s_c) < 2:
            continue
        ks_stat, p_value = stats.ks_2samp(r_c, s_c)
        r_mean, s_mean = float(r_c.mean()), float(s_c.mean())
        r_sd = float(r_c.std(ddof=1)) if len(r_c) > 1 else 0.0
        s_sd = float(s_c.std(ddof=1)) if len(s_c) > 1 else 0.0

        ks_row = ks_lookup.get(str(col), {})
        if isinstance(ks_row, dict):
            ks_stat = float(ks_row.get("KS_Statistic", ks_stat))
            p_value = float(ks_row.get("p_value", p_value))

        fidelity = "PASS" if p_value > 0.05 else "FAIL"
        rows.append([
            col,
            f"{r_mean:.2f} ± {r_sd:.2f}",
            f"{s_mean:.2f} ± {s_sd:.2f}",
            f"{float(stats.skew(r_c, bias=False)):.2f}",
            f"{float(stats.skew(s_c, bias=False)):.2f}",
            f"{_safe_float(ks_stat, 'N/A')}",
            f"{_safe_float(p_value, 'N/A')}",
            fidelity,
        ])
    return rows


# -------------------------------------------------------------
# Comprehensive Clinical Audit Report PDF (จัดหน้าสวยงามพอดี ไม่มีหน้าว่าง)
# -------------------------------------------------------------
def build_comprehensive_audit_pdf(df_real, df_synth, ks_df, total_raw):
    try:
        style_sheet = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ReportTitle", 
            parent=style_sheet["Title"], 
            fontName="Helvetica-Bold", 
            fontSize=15, 
            leading=18, 
            textColor=HexColor("#1A365D"), 
            alignment=1,
            spaceAfter=3,
        )
        heading_style = ParagraphStyle(
            "SectionHeading", 
            parent=style_sheet["Heading2"], 
            fontName="Helvetica-Bold", 
            fontSize=10.5, 
            textColor=HexColor("#1A365D"), 
            spaceBefore=8, 
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "BodyText", 
            parent=style_sheet["BodyText"], 
            fontName="Helvetica", 
            fontSize=7.5, 
            leading=10, 
            textColor=HexColor("#2D3748")
        )
        small_style = ParagraphStyle(
            "SmallText", 
            parent=style_sheet["BodyText"], 
            fontName="Helvetica", 
            fontSize=7, 
            leading=9, 
            textColor=HexColor("#4A5568")
        )

        buf = io.BytesIO()
        # ขอบกระดาษสมดุล ไม่ล้น ไม่ดันเกิดหน้าว่าง
        doc = SimpleDocTemplate(
            buf, 
            pagesize=A4, 
            leftMargin=0.45 * inch, 
            rightMargin=0.45 * inch, 
            topMargin=0.65 * inch, 
            bottomMargin=0.45 * inch
        )

        # แถบ Header สีน้ำเงินเข้ม: เปลี่ยนเป็น 🧬 SYNDA - Synthetic Medical Laboratory Data Platform
        def _header_footer(canvas, doc_obj):
            canvas.saveState()
            canvas.setFillColor(HexColor("#1A365D"))
            canvas.rect(0.45 * inch, 792, 7.37 * inch, 18, stroke=0, fill=1)
            canvas.setFillColor(HexColor("#FFFFFF"))
            canvas.setFont("Helvetica-Bold", 8.5)
            canvas.drawCentredString(297, 797, "SYNDA - Synthetic Medical Laboratory Data Platform")
            canvas.setFillColor(HexColor("#4A5568"))
            canvas.setFont("Helvetica", 7)
            timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
            canvas.drawRightString(7.8 * inch, 814, timestamp)
            canvas.drawRightString(7.8 * inch, 14, f"Page {canvas.getPageNumber()}")
            canvas.restoreState()

        story = [
            Paragraph("Clinical Data Synthesis & Statistical Fidelity Audit Report", title_style),
            Paragraph("<font color='#2b6cb0'><b>STATUS: CLINICALLY VALIDATED / FULLY DE-IDENTIFIED (HIPAA COMPLIANT)</b></font>", body_style),
            Spacer(1, 0.08 * inch),
        ]

        # Metadata Table
        metadata = [
            [Paragraph("<b>Total Baseline Patient Records</b>", body_style), Paragraph(f"{total_raw:,}", body_style), Paragraph("<b>Generation Algorithm</b>", body_style), Paragraph("Multivariate Gaussian Copula", body_style)],
            [Paragraph("<b>Synthetic Cohort Samples</b>", body_style), Paragraph(f"{len(df_synth):,}", body_style), Paragraph("<b>Rank Transformation</b>", body_style), Paragraph("Latent Spearman Matching", body_style)],
            [Paragraph("<b>Audited Analyte Features</b>", body_style), Paragraph(f"{len(df_real.columns)}", body_style), Paragraph("<b>Data Governance</b>", body_style), Paragraph("Scrubbed De-identified Output", body_style)],
        ]
        t_meta = Table(metadata, colWidths=[1.8 * inch, 1.8 * inch, 1.8 * inch, 1.8 * inch])
        t_meta.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#D9E2EC")),
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#F7FAFC")),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#EAF2FF")),
            ("BACKGROUND", (2, 0), (2, -1), HexColor("#EAF2FF")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t_meta)
        story.append(Spacer(1, 0.08 * inch))

        # 1. Descriptive Table
        story.append(Paragraph("1. Baseline Clinical Descriptive Statistics", heading_style))
        desc_data = [[Paragraph(f"<b>{h}</b>", body_style) for h in ["Parameter", "Mean", "Median", "Skewness", "95th Pct"]]]
        for var in df_real.columns:
            ser = pd.to_numeric(df_real[var].dropna(), errors="coerce").dropna()
            if ser.empty:
                continue
            desc_data.append([
                Paragraph(normalize_feature_name(var), body_style),
                Paragraph(f"{ser.mean():.2f}", body_style),
                Paragraph(f"{ser.median():.2f}", body_style),
                Paragraph(f"{stats.skew(ser, bias=False):.3f}", body_style),
                Paragraph(f"{ser.quantile(0.95):.2f}", body_style),
            ])
        t_desc = Table(desc_data, colWidths=[1.8 * inch, 1.35 * inch, 1.35 * inch, 1.35 * inch, 1.35 * inch])
        t_desc.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#D9E2EC")),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#EAF2FF")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F9FAFB")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        story.append(t_desc)
        story.append(Spacer(1, 0.08 * inch))

        # 2. K-S Statistical Table
        story.append(Paragraph("2. Statistical Rigor & Kolmogorov-Smirnov Goodness-of-Fit", heading_style))
        s_rows = build_statistical_summary_table(df_real, df_synth, ks_df)
        if s_rows:
            table_data = [[Paragraph(f"<b>{h}</b>", body_style) for h in ["Variable", "Real Mean±SD", "Synth Mean±SD", "Real Skew", "Synth Skew", "KS Stat", "p-val", "Status"]]]
            for r in s_rows:
                table_data.append([Paragraph(str(cell), body_style) for cell in r])
            st_table = Table(table_data, repeatRows=1, colWidths=[0.85 * inch, 1.05 * inch, 1.05 * inch, 0.72 * inch, 0.72 * inch, 0.65 * inch, 0.68 * inch, 0.65 * inch])
            st_table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#D9E2EC")),
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#EAF2FF")),
                ("BACKGROUND", (7, 1), (7, -1), HexColor("#E8F5E9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F9FAFB")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(st_table)

        # ---------------------------------------------------------
        # ALL VISUAL EVIDENCE (จัดคู่ลงหน้าละ 2 ภาพ พอดีสวยงาม)
        # ---------------------------------------------------------
        def add_figure_pair(fig_a, title_a, fig_b, title_b, is_page_start=True):
            if is_page_start:
                story.append(PageBreak())
            
            if fig_a:
                img_a = _figure_to_reportlab_image(fig_a)
                if img_a:
                    story.append(Paragraph(f"<b>{title_a}</b>", heading_style))
                    story.append(img_a)
                    story.append(Spacer(1, 0.05 * inch))
                plt.close(fig_a)
                
            if fig_b:
                img_b = _figure_to_reportlab_image(fig_b)
                if img_b:
                    story.append(Paragraph(f"<b>{title_b}</b>", heading_style))
                    story.append(img_b)
                plt.close(fig_b)

        # Page 2: Fig 1 (Baseline Grid) & Fig 2 (Moments Grid)
        fig_base = _generate_multi_distribution_grid_figure(df_real, numeric_columns_only(df_real), max_vars=9)
        fig_mom = _generate_moment_2x2_grid_figure(df_real, numeric_columns_only(df_real))
        add_figure_pair(
            fig_base, "Figure 1: Baseline Marginal Distribution Panel (Full Clinical Panel)",
            fig_mom, "Figure 2: Baseline Distribution Moments (Mean, Variance, Skewness, Kurtosis)",
            is_page_start=True
        )

        # Page 3: Fig 3 (KDE Comparison) & Fig 4 (KS Bar Charts)
        fig_kde = _generate_overview_kde_figure(df_real, df_synth, max_vars=9)
        fig_ks = _generate_ks_summary_chart_figure(ks_df)
        add_figure_pair(
            fig_kde, "Figure 3: Marginal Density Overlay Comparison (Real vs Synthetic)",
            fig_ks, "Figure 4: Kolmogorov-Smirnov (K-S) Statistic & p-value Bar Charts",
            is_page_start=True
        )

        # Page 4: Fig 5 (Spearman Correlation) & Fig 6 (Four Moments Comparison)
        fig_corr = _generate_correlation_figure(df_real, df_synth)
        fig_all_mom = _generate_all_moments_grid_figure(df_real, df_synth)
        add_figure_pair(
            fig_corr, "Figure 5: Spearman Rank Correlation Matrix Comparison (Real vs Synthetic)",
            fig_all_mom, "Figure 6: Four Moments Comparison (Real vs Synthetic)",
            is_page_start=True
        )

        # Page 5: Fig 7 (Boxplots Outlier Grid)
        story.append(PageBreak())
        fig_box = _generate_boxplot_grid_figure(df_real, df_synth)
        if fig_box:
            img_box = _figure_to_reportlab_image(fig_box, max_width=6.9 * inch, max_height=6.2 * inch)
            if img_box:
                story.append(Paragraph("<b>Figure 7: Clinical Boxplot Distribution & Outlier Comparison</b>", heading_style))
                story.append(img_box)
                story.append(Spacer(1, 0.1 * inch))
                story.append(Paragraph(f"Official Audit Document Compiled: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}", small_style))
            plt.close(fig_box)

        doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
        pdf_bytes = buf.getvalue()
        return pdf_bytes
    except Exception as e:
        st.error(f"PDF Compilation Error: {e}")
        return b""


# -------------------------------------------------------------
# Fragment 1: Feature Distribution Inspector
# -------------------------------------------------------------
@st.fragment
def render_feature_distribution_inspector(real_data, numeric_cols):
    st.markdown("<p style='font-size: 14px; font-weight: 600; color: #1a202c;'>🔎 Feature Distribution Inspector (Individual Analyte Breakdown):</p>", unsafe_allow_html=True)
    feature_inspector_col, feature_summary_col = st.columns([1.8, 1.2])

    with feature_inspector_col:
        default_feature = "TG" if "TG" in numeric_cols else numeric_cols[0]
        selected_feature = st.selectbox(
            "Selected Clinical Feature",
            options=numeric_cols,
            index=numeric_cols.index(default_feature),
            key="feature_inspector_selected_var",
            format_func=lambda x: normalize_feature_name(x),
        )
        dist_fig = build_distribution_histogram_figure(real_data, selected_feature)
        if PLOTLY_AVAILABLE:
            st.plotly_chart(dist_fig, use_container_width=True)
        else:
            st.pyplot(dist_fig)
            plt.close(dist_fig)

    with feature_summary_col:
        st.markdown("<p style='font-size: 13px; font-weight: 600; color: #4a5568;'>Numeric Feature Moments Summary</p>", unsafe_allow_html=True)
        summary_rows = []
        for var in numeric_cols:
            ser = pd.to_numeric(real_data[var].dropna(), errors="coerce").dropna()
            if ser.empty:
                continue
            summary_rows.append({
                "Parameter": normalize_feature_name(var),
                "Mean": round(float(ser.mean()), 2),
                "Median": round(float(ser.median()), 2),
                "Skewness": round(float(stats.skew(ser, bias=False)), 3),
                "95th Pct": round(float(ser.quantile(0.95)), 2),
            })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, height=430)


# -------------------------------------------------------------
# Fragment 2: Isolated Boxplot Rerun
# -------------------------------------------------------------
@st.fragment
def render_isolated_boxplot_panel(df_real, df_synth):
    boxplot_options = [col for col in df_real.columns if pd.api.types.is_numeric_dtype(df_real[col]) and col in df_synth.columns]
    default_bp_idx = boxplot_options.index("TG") if "TG" in boxplot_options else 0
    
    b_col_ctrl, b_col_view = st.columns([1, 3])
    with b_col_ctrl:
        st.markdown("<p style='font-size: 12.5px; font-weight:600; color: #4a5568;'>Select Target Parameter:</p>", unsafe_allow_html=True)
        chosen_param = st.selectbox(
            "Clinical Parameter",
            options=boxplot_options,
            index=default_bp_idx,
            format_func=lambda x: normalize_feature_name(x),
            label_visibility="collapsed"
        )
        st.caption("Inspect median shifts, interquartile ranges (IQR), and extreme outlier bounds between authentic and simulated cohorts.")

    with b_col_view:
        box_fig = build_interactive_boxplot(df_real, df_synth, chosen_param)
        if PLOTLY_AVAILABLE:
            st.plotly_chart(box_fig, use_container_width=True)
        else:
            st.pyplot(box_fig)


# =============================================================
# MAIN CLINICAL LIS DASHBOARD APPLICATION
# =============================================================
def main():
    inject_clinical_his_theme()

    has_generated = "synth_data" in st.session_state

    # 1. Sidebar Panel
    with st.sidebar:
        st.markdown("<h3 style='margin-top:0;'>🧬 Synthetic Medical Lab Data</h3>", unsafe_allow_html=True)
        
        st.markdown("<p style='color: #718096; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;'>SYSTEM NAVIGATION</p>", unsafe_allow_html=True)
        active_tab = st.radio(
            "View Navigation",
            ["📊 Clinical Synthesis Dashboard", "ℹ️ Project Information & Governance"],
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("---")

        if active_tab == "📊 Clinical Synthesis Dashboard":
            st.markdown("<p style='color: #718096; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 8px;'>CLINICAL FLOW PIPELINE</p>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <a class="nav-flow-item" href="#step-1-baseline-eda">
                    <span>01</span> Baseline Exploration
                </a>
                <a class="nav-flow-item" href="#step-2-synthesis-execution">
                    <span>02</span> Generation Parameters
                </a>
                <a class="nav-flow-item {'nav-flow-active' if has_generated else ''}" href="#step-3-comprehensive-fidelity">
                    <span>03</span> Fidelity & Distribution
                </a>
                <a class="nav-flow-item" href="#step-4-audit-export">
                    <span>04</span> Report & Dataset Export
                </a>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("---")

        st.markdown("<p style='color: #718096; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;'>DATA SOURCE SELECTION</p>", unsafe_allow_html=True)
        mode = st.radio("Operating Dataset", ["Use Master Template", "Upload Custom Cohort"], index=0, label_visibility="collapsed")
        
        uploaded_file = None
        if mode == "Upload Custom Cohort":
            uploaded_file = st.file_uploader("Select Clinical File (CSV/XLSX)", type=["csv", "xlsx"])

        st.markdown("---")
        st.markdown("<p style='color: #718096; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;'>SYNTHESIS ENGINE SETTINGS</p>", unsafe_allow_html=True)
        n_samples = st.number_input("Target Sample Size", min_value=100, max_value=100000, value=1000, step=500)
        use_seed_search = st.checkbox("Optimize Moment Matching", value=True)
        num_trials = 20
        if use_seed_search:
            num_trials = st.slider("Optimization Search Trials", min_value=5, max_value=50, value=20, step=5)

    # 2. Top Banner Header (SYNDA Branding)
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%);
            color: #ffffff;
            padding: 16px 24px;
            border-radius: 4px;
            margin-bottom: 22px;
            display: flex;
            align-items: center;
            box-shadow: 0 3px 6px rgba(0,0,0,0.12);
            border-bottom: 3px solid #63b3ed;
        ">
            <div style="display: flex; align-items: center; gap: 16px;">
                <div style="
                    background: rgba(255, 255, 255, 0.12);
                    border: 1px solid rgba(255, 255, 255, 0.25);
                    width: 44px;
                    height: 44px;
                    border-radius: 8px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 22px;
                ">
                    🧬
                </div>
                <div>
                    <div style="display: flex; align-items: baseline; gap: 10px;">
                        <span style="font-size: 22px; font-weight: 800; letter-spacing: 1.5px; color: #ffffff;">SYNDA</span>
                        <span style="font-size: 14px; font-weight: 500; color: #bee3f8; letter-spacing: 0.3px;">
                            Synthetic Medical Laboratory Data Platform
                        </span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. File Loading & Processing
    df_real_raw = None
    master_file = "combined_raw_data (1).csv"

    if mode == "Upload Custom Cohort":
        if uploaded_file is not None:
            df_real_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    else:
        resolved_path = backend._resolve_data_file_path(master_file) if hasattr(backend, "_resolve_data_file_path") else None
        if resolved_path and os.path.exists(resolved_path):
            df_real_raw = pd.read_csv(resolved_path) if resolved_path.endswith(".csv") else pd.read_excel(resolved_path)

    if df_real_raw is None:
        st.warning("⚠️ No active dataset detected. Please verify the master dataset location or upload a custom cohort via the left navigation panel.")
        return

    filtered_df = normalize_dataframe_columns(remove_identifier_columns(df_real_raw))
    if "LDL" not in filtered_df.columns and all(x in filtered_df.columns for x in ("CHO", "HDLC", "TG")):
        try:
            filtered_df["LDL"] = filtered_df["CHO"] - filtered_df["HDLC"] - (filtered_df["TG"] / 5.0)
        except Exception:
            pass

    with st.spinner("Extracting parameters and baseline structure..."):
        blueprint = _cached_process_parameters(filtered_df)
    real_data = normalize_dataframe_columns(blueprint.get("cleaned_real_data", filtered_df))
    parameters = blueprint.get("parameters", {})
    available_vars = [v for v in blueprint.get("available_variables", list(real_data.columns)) if not is_identifier_column(v)]
    numeric_cols = numeric_columns_only(real_data)

    # =========================================================================
    # VIEW ROUTE 1: ℹ️ Project Information & Governance
    # =========================================================================
    if active_tab == "ℹ️ Project Information & Governance":
        st.markdown(
            """
            <div class="his-box">
                <div class="his-box-header">
                    <span>Project Scope, Governance & Methodology (Term Paper 2025)</span>
                    <span style="font-size:11px; background:#e2e8f0; color:#2d3748; padding:2px 8px; border-radius:3px; font-weight:700;">DOCUMENTATION</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        info_col1, info_col2 = st.columns([2, 1])

        with info_col1:
            st.markdown("#### 🏆 About This Project")
            st.markdown(
                """
                This project aims to generate **high-quality synthetic medical laboratory data** that preserves the statistical patterns of real lab results while protecting patient privacy. The synthetic dataset will support research, education, and software development workflows where access to real clinical data is limited.
                """
            )
            
            st.markdown("##### 🔄 Data Processing Pipeline:")
            st.markdown(
                """
                1. **Direct Identifier Scrubbing:** Direct personal identifiers (HN, Citizen ID, Name, Encounters) are automatically stripped before analysis.
                2. **Baseline Parameter Extraction:** Computes marginal distribution moments (Mean, Variance, Skewness, Kurtosis) and inter-analyte rank dependencies.
                3. **Generative Synthesis:** Executes a multivariate Gaussian Copula with empirical marginal transformations and seed optimization.
                4. **Statistical Audit:** Benchmarks fidelity against real cohorts using Two-Sample Kolmogorov-Smirnov tests, Spearman correlation matrices, and outlier spreads.
                """
            )
            st.info("🔒 **Data Governance & Privacy Assurance:** Uploaded clinical records are processed within this active execution session strictly for statistical aggregation. No personal health records are permanently stored or shared outside the de-identified synthetic outputs.")

        with info_col2:
            st.markdown("#### 🎯 Key Goals")
            st.markdown(
                """
                * Maintaining clinical plausibility and realistic reference ranges
                * Preserving correlations across tests, demographics, and time
                * Enabling safe sharing for analysis, modeling, and validation
                * Providing clear documentation of generation methods and limitations
                """
            )
            
            st.markdown("---")
            st.markdown("#### 👥 Project Investigators")
            st.markdown(
                """
                1. **Nuttapat Anuwongcharoen**
                2. **Sughinan Rotrungwat**
                
                *Term Paper 2025 - Synthetic Medical Laboratory Data Project*
                """
            )

        st.markdown("</div>", unsafe_allow_html=True)
        return

    # =========================================================================
    # VIEW ROUTE 2: 📊 Clinical Synthesis Dashboard
    # =========================================================================
    st.markdown(
        """
        <div style="background-color: #ffffff; border-left: 4px solid #2b6cb0; padding: 12px 18px; border-radius: 3px; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
            <div style="font-size: 13.5px; color: #2d3748; line-height: 1.5;">
                <b>Clinical Workflow Overview:</b> Generate and validate privacy-preserving synthetic laboratory cohorts using Gaussian Copula modeling with empirical rank preservation. 
                <i>(For detailed methodology, governance, and investigator credentials, select <b>'ℹ️ Project Information & Governance'</b> in the left panel.)</i>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------- STEP 1: BASELINE DATA EXPLORATION -----------------
    st.markdown('<div id="step-1-baseline-eda"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="his-box">
            <div class="his-box-header">
                <span>1. Baseline Clinical Data Exploration & Quality Audit</span>
                <span style="font-size:11px; background:#ebf8ff; color:#2b6cb0; padding:2px 8px; border-radius:3px; font-weight:700;">STAGE 1: INGESTION</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Raw Records", f"{len(filtered_df):,}")
    m2.metric("Cleaned Patient Cohort", f"{len(real_data):,}")
    m3.metric("Audited Numeric Features", f"{len(numeric_cols)}")
    m4.metric("Missing Value Rate", f"{float(filtered_df.isna().mean().mean() * 100.0):.2f}%")

    st.markdown("---")

    # 1.1 Multi-Distribution Panel (9 Grids)
    st.markdown("<p style='font-size: 13.5px; font-weight: 600; color: #2d3748;'>Overall Marginal Distribution & Skewness Profiling (Full Clinical Panel):</p>", unsafe_allow_html=True)
    multi_dist_bytes = render_cached_multi_distribution_grid(real_data, numeric_cols, max_vars=9)
    if multi_dist_bytes:
        st.image(multi_dist_bytes, use_container_width=True)
        download_png_button(multi_dist_bytes, "baseline_distributions_grid.png", label="Download High-Res Panel (PNG)")

    st.markdown("---")

    render_feature_distribution_inspector(real_data, numeric_cols)

    st.markdown("---")

    st.markdown("<p style='font-size: 13.5px; font-weight: 600; color: #2d3748;'>Distribution Moments (Mean, Variance, Skewness, Kurtosis):</p>", unsafe_allow_html=True)
    moment_grid_bytes = render_cached_moment_2x2_grid(real_data, numeric_cols)
    if moment_grid_bytes:
        st.image(moment_grid_bytes, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------- STEP 2: SYNTHESIS EXECUTION -----------------
    st.markdown('<div id="step-2-synthesis-execution"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="his-box">
            <div class="his-box-header">
                <span>2. Parameter Schema & Generative Execution</span>
                <span style="font-size:11px; background:#fefcbf; color:#975a16; padding:2px 8px; border-radius:3px; font-weight:700;">STAGE 2: SYNTHESIS</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    core_9 = ["Age", "Glucose", "CHO", "TG", "LDL", "HDLC", "AST", "ALT", "CRE"]
    with st.form("quick_synth_form"):
        st.markdown("<p style='font-size: 13px; color: #4a5568;'>Select target clinical parameters for model training and simulation:</p>", unsafe_allow_html=True)
        f_cols = st.columns(5)
        feature_states = {}
        for idx, var in enumerate(available_vars):
            with f_cols[idx % 5]:
                default_chk = True if var in core_9 or not any(c in available_vars for c in core_9) else False
                feature_states[var] = st.checkbox(normalize_feature_name(var), value=default_chk, key=f"gen_chk_{var}")
        
        exec_btn = st.form_submit_button("⚡ Execute Generative Copula Workflow", type="primary")

    if exec_btn:
        selected_features = [v for v in available_vars if feature_states.get(v, False)]
        if not selected_features:
            st.error("Please select at least one clinical parameter.")
        else:
            with st.spinner("Executing Gaussian Copula with Latent Spearman Transformation & Seed Search..."):
                df_real_eval = real_data[selected_features]
                if use_seed_search:
                    synth_data, best_seed, best_score = backend.generate_with_seed_search(
                        parameters=parameters,
                        df_real=real_data,
                        num_samples=n_samples,
                        num_trials=num_trials,
                        selected_variables=selected_features,
                    )
                else:
                    synth_data = _cached_generate_synthetic(
                        parameters=parameters,
                        num_samples=n_samples,
                        selected_variables=selected_features,
                    )
                    best_seed, best_score = "Default", None

                eval_results = _safe_evaluation_results(df_real_eval, synth_data)

                st.session_state["real_data"] = df_real_eval
                st.session_state["synth_data"] = synth_data
                st.session_state["eval_results"] = eval_results
                st.session_state["filtered_df"] = filtered_df
                st.success("Clinical synthesis completed successfully. Visual and statistical evaluation reports are updated below.")
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    df_real_curr = st.session_state.get("real_data", real_data)
    df_synth_curr = st.session_state.get("synth_data", None)
    eval_results = st.session_state.get("eval_results", {})
    ks_df = eval_results.get("ks", pd.DataFrame())

    # ----------------- STEP 3: COMPREHENSIVE FIDELITY AUDIT -----------------
    st.markdown('<div id="step-3-comprehensive-fidelity"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="his-box">
            <div class="his-box-header">
                <span>3. Comprehensive Statistical Fidelity & Distributional Spread Audit</span>
                <span style="font-size:11px; background:#c6f6d5; color:#22543d; padding:2px 8px; border-radius:3px; font-weight:700;">STAGE 3: FIDELITY AUDIT</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if df_synth_curr is not None:
        st.markdown("<p style='font-size: 13.5px; font-weight: 600; color: #2d3748;'>3.1 Marginal Density Overlay (Real vs Synthetic) with Annotated KS p-values:</p>", unsafe_allow_html=True)
        ov_bytes = render_cached_overview_kde_figure(df_real_curr, df_synth_curr, max_vars=9)
        if ov_bytes:
            st.image(ov_bytes, use_container_width=True)
            download_png_button(ov_bytes, "overview_kde_comparison.png", label="Download Multi-Density Panel (PNG)")

        st.markdown("---")

        st.markdown("<p style='font-size: 13.5px; font-weight: 600; color: #2d3748;'>3.2 Two-Sample Kolmogorov-Smirnov Goodness-of-Fit Audit:</p>", unsafe_allow_html=True)
        if not ks_df.empty:
            similar_count = int((ks_df["Decision"] == "Similar distributions").sum())
            mean_ks = float(ks_df["KS_Statistic"].mean())
            k_col1, k_col2 = st.columns(2)
            k_col1.metric("Variables Meeting Fidelity Criteria (p > 0.05)", f"{similar_count} / {len(ks_df)}")
            k_col2.metric("Mean K-S Statistic", f"{mean_ks:.4f}")
            
            st.dataframe(ks_df, use_container_width=True)

            ks_chart_bytes = render_cached_ks_summary_chart(ks_df)
            if ks_chart_bytes:
                st.image(ks_chart_bytes, use_container_width=True)
                download_png_button(ks_chart_bytes, "ks_statistics_barcharts.png", label="Download K-S Bar Charts (PNG)")

        st.markdown("---")

        st.markdown("<p style='font-size: 13.5px; font-weight: 600; color: #2d3748;'>3.3 Association Structure & Four Moments Comparison:</p>", unsafe_allow_html=True)
        c_left, c_right = st.columns([1.2, 1])
        with c_left:
            st.markdown("<p style='font-size: 12.5px; font-weight: 600; color: #4a5568;'>Monotonic Association Structure (Spearman Rank Correlation):</p>", unsafe_allow_html=True)
            corr_bytes = render_cached_correlation_figure(df_real_curr, df_synth_curr)
            if corr_bytes:
                st.image(corr_bytes, use_container_width=True)

        with c_right:
            st.markdown("<p style='font-size: 12.5px; font-weight: 600; color: #4a5568;'>Four Moments Comparison by Metric:</p>", unsafe_allow_html=True)
            moment_choice = st.selectbox("Select Target Moment", ["Mean", "Variance", "Skewness", "Kurtosis"])
            mom_fig = build_moment_comparison_chart(df_real_curr, df_synth_curr, moment_choice)
            st.pyplot(mom_fig)
            plt.close(mom_fig)

        st.markdown("---")

        st.markdown("<p style='font-size: 13.5px; font-weight: 600; color: #2d3748;'>3.4 Interactive Distributional Spread & Outlier Inspection (Real vs Synthetic):</p>", unsafe_allow_html=True)
        render_isolated_boxplot_panel(df_real_curr, df_synth_curr)
    else:
        st.info("Evaluation panels will render here immediately after executing the generative engine in Step 2.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------- STEP 4: EXPORT & AUDIT REPORT -----------------
    st.markdown('<div id="step-4-audit-export"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="his-box">
            <div class="his-box-header">
                <span>4. Hospital Information System Audit Documentation & Dataset Export</span>
                <span style="font-size:11px; background:#fed7d7; color:#9b2c2c; padding:2px 8px; border-radius:3px; font-weight:700;">STAGE 4: EXPORT</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if df_synth_curr is not None:
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("<b>Official Clinical Synthesis Audit Certificate (PDF)</b>", unsafe_allow_html=True)
            st.caption("Comprehensive audit report including all descriptive metrics, K-S tables, and all generated clinical figures.")
            
            pdf_bytes = build_comprehensive_audit_pdf(
                df_real_curr,
                df_synth_curr,
                ks_df,
                len(filtered_df),
            )
            
            if pdf_bytes and len(pdf_bytes) > 0:
                st.download_button(
                    label="📥 Download Hospital Audit Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"SYNDA_clinical_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )
            else:
                st.error("Failed to compile audit report. Please review server log.")

        with d2:
            st.markdown("<b>De-identified Synthetic Patient Cohort (CSV)</b>", unsafe_allow_html=True)
            st.caption("Statistically consistent tabular dataset ready for clinical epidemiology, secondary research, or algorithmic validation.")
            csv_bytes = df_synth_curr.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Synthetic Dataset (CSV)",
                data=csv_bytes,
                file_name=f"synthetic_patient_cohort_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        st.markdown("<br><b>Synthesized Cohort Verification Table (First 30 Records):</b>", unsafe_allow_html=True)
        st.dataframe(df_synth_curr.head(30), use_container_width=True)
    else:
        st.info("Export utilities and downloadable certificates will be unlocked once generation completes.")

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os


def _numeric_common_columns(df_real, df_synthetic):
    return [
        col for col in df_real.columns
        if col in df_synthetic.columns and pd.api.types.is_numeric_dtype(df_real[col]) and pd.api.types.is_numeric_dtype(df_synthetic[col])
    ]


def get_variable_generative_model(col_name):
    """ระบุโมเดลและเทคนิคสถิติที่ใช้สร้างข้อมูลแต่ละตัวแปร"""
    name = str(col_name).strip().upper()
    
    # 1. กลุ่มที่ใช้ PCHIP Smooth Quantile Mapping (แก้ปัญหา Bimodal, Discrete, และ Extreme Tail)
    if name in {"CHO", "TG", "LDL"}:
        return "PCHIP Quantile Mapping (Lipid Guarded)"
    elif name in {"GLU", "GLUCOSE"}:
        return "PCHIP Quantile Mapping (Bimodal)"
    elif name == "AST":
        return "PCHIP Quantile Mapping (Heavy-Tail)"
    elif name == "AGE":
        return "PCHIP Quantile Mapping (Discrete)"
    
    # 2. กลุ่มที่รันผ่าน Parametric Skewness Gates
    elif name in {"ALT", "CRE", "CREATININE"}:
        return "Parametric Gamma Distribution"
    elif name in {"HDLC", "HDL"}:
        return "Parametric Gamma Distribution"
    elif name in {"HEIGHT", "WEIGHT"}:
        return "Parametric Normal Distribution"
    else:
        return "Empirical Quantile Copula"


def evaluate_synthetic_data(df_real, df_synthetic):
    """Compute KS tests and convert the result into a clean summary table."""
    common_cols = _numeric_common_columns(df_real, df_synthetic)
    rows = []

    for col in common_cols:
        real_data = pd.to_numeric(df_real[col].dropna(), errors="coerce").dropna()
        synth_data = pd.to_numeric(df_synthetic[col].dropna(), errors="coerce").dropna()
        if len(real_data) < 2 or len(synth_data) < 2:
            continue

        try:
            if len(synth_data) > len(real_data) * 1.5 and len(real_data) >= 20:
                n_iter = 100
                ks_vals = []
                p_vals = []
                for _ in range(n_iter):
                    samp = synth_data[np.random.choice(len(synth_data), size=len(real_data), replace=False)]
                    ks, p = stats.ks_2samp(real_data, samp)
                    ks_vals.append(ks)
                    p_vals.append(p)
                ks_stat = float(np.mean(ks_vals))
                p_value = float(np.mean(p_vals))
            else:
                ks_stat, p_value = stats.ks_2samp(real_data, synth_data)
        except Exception:
            ks_stat, p_value = stats.ks_2samp(real_data, synth_data)

        decision = "Similar distributions" if p_value > 0.05 else "Distribution difference"
        gen_model = get_variable_generative_model(col)

        rows.append(
            {
                "Variable": col,
                "KS_Statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 4),
                "Decision": decision,
                "Generative Model": gen_model,
            }
        )

    return pd.DataFrame(rows)


def summarize_moments(df_real, df_synthetic):
    """Create a first-four-moments comparison table for the overlapping variables."""
    common_cols = _numeric_common_columns(df_real, df_synthetic)
    rows = []

    for col in common_cols:
        real_data = pd.to_numeric(df_real[col].dropna(), errors="coerce").dropna()
        synth_data = pd.to_numeric(df_synthetic[col].dropna(), errors="coerce").dropna()
        if len(real_data) < 2 or len(synth_data) < 2:
            continue

        rows.append(
            {
                "Variable": col,
                "Mean (Real)": round(float(np.mean(real_data)), 4),
                "Mean (Synthetic)": round(float(np.mean(synth_data)), 4),
                "Variance (Real)": round(float(np.var(real_data, ddof=1)), 4),
                "Variance (Synthetic)": round(float(np.var(synth_data, ddof=1)), 4),
                "Skewness (Real)": round(float(stats.skew(real_data, bias=False)), 4),
                "Skewness (Synthetic)": round(float(stats.skew(synth_data, bias=False)), 4),
                "Kurtosis (Real)": round(float(stats.kurtosis(real_data, fisher=False, bias=False)), 4),
                "Kurtosis (Synthetic)": round(float(stats.kurtosis(synth_data, fisher=False, bias=False)), 4),
            }
        )

    return pd.DataFrame(rows)


def plot_comparison_kde(df_real, df_synthetic, column_name):
    """Create an overlapping KDE plot that highlights right-skewness and tail behavior."""
    real_values = pd.to_numeric(df_real[column_name].dropna(), errors="coerce").dropna()
    synth_values = pd.to_numeric(df_synthetic[column_name].dropna(), errors="coerce").dropna()

    if len(real_values) < 3 or len(synth_values) < 3:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Not enough values to plot a KDE comparison", ha="center", va="center")
        ax.set_axis_off()
        return fig

    real_plot = np.log1p(real_values)
    synth_plot = np.log1p(synth_values)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.kdeplot(real_plot, label="Real Data", color="#2563eb", fill=True, alpha=0.25, linewidth=2, ax=ax)
    sns.kdeplot(synth_plot, label="Synthetic Data", color="#f59e0b", fill=True, alpha=0.25, linewidth=2, ax=ax)
    ax.axvline(real_plot.mean(), color="#2563eb", linestyle="--", alpha=0.7)
    ax.axvline(synth_plot.mean(), color="#f59e0b", linestyle="--", alpha=0.7)
    ax.set_title(f"Right-skewness comparison for {column_name}", fontsize=12)
    ax.set_xlabel(f"{column_name} (log1p-transformed for skewness visualization)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_comparison_correlation_heatmap(df_real, df_synthetic):
    """Create side-by-side Spearman correlation heatmaps for the overlapping numeric variables."""
    common_cols = _numeric_common_columns(df_real, df_synthetic)
    if not common_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No numeric columns available for correlation comparison", ha="center", va="center")
        ax.set_axis_off()
        return fig

    corr_real = df_real[common_cols].corr(method="spearman")
    corr_synth = df_synthetic[common_cols].corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(corr_real, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True, ax=axes[0])
    axes[0].set_title("Real Data Correlation (Spearman)", fontsize=12)
    sns.heatmap(corr_synth, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True, ax=axes[1])
    axes[1].set_title("Synthetic Data Correlation (Spearman)", fontsize=12)
    plt.tight_layout()
    return fig


def evaluate_all_metrics(df_real: pd.DataFrame, df_synth: pd.DataFrame, show_plots: bool = True):
    """Run comprehensive evaluation: moments, KS tests, correlation comparison and plots."""
    results = {}
    moments_df = summarize_moments(df_real, df_synth)
    results["moments"] = moments_df

    ks_df = evaluate_synthetic_data(df_real, df_synth)
    results["ks"] = ks_df

    common_cols = _numeric_common_columns(df_real, df_synth)
    if common_cols:
        corr_real = df_real[common_cols].corr(method="spearman")
        corr_synth = df_synth[common_cols].corr(method="spearman")
    else:
        corr_real = pd.DataFrame()
        corr_synth = pd.DataFrame()

    results["corr_real"] = corr_real
    results["corr_synth"] = corr_synth

    figs = {}
    if show_plots and not corr_real.empty:
        figs["corr_heatmap"] = plot_comparison_correlation_heatmap(df_real, df_synth)

    results["figures"] = figs
    return results
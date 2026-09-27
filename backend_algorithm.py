import os
import time
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def process_and_extract_parameters(df_input: pd.DataFrame):
    blueprint_vars = ["Age", "Weight", "Sex", "Glucose", "CHO", "TG", "LDL", "HDLC", "AST", "ALT", "CRE"]

    column_mapping = {
        "GLU": "Glucose",
        "FBS": "Glucose",
        "BS": "Glucose",
        "CHOL": "CHO",
        "TOTAL_CHOLESTEROL": "CHO",
        "TRIGLYCERIDE": "TG",
        "TRIGLYCERIDES": "TG",
        "HDL": "HDLC",
        "HDL-C": "HDLC",
        "LDL-C": "LDL",
        "CREATININE": "CRE",
    }

    df = df_input.copy()
    df.columns = df.columns.str.strip()
    df.rename(columns=column_mapping, inplace=True)

    exclude_cols = ["HN", "ID", "Name", "Firstname", "Lastname", "ServiceDate"]
    df = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")

    available_vars = [v for v in blueprint_vars if v in df.columns]
    if not available_vars:
        return {
            "status": "success",
            "cleaned_data": pd.DataFrame(),
            "cleaned_real_data": pd.DataFrame(),
            "parameters": {},
            "available_variables": [],
        }

    coerced = df[available_vars].apply(pd.to_numeric, errors="coerce")
    nunique = df[available_vars].apply(lambda s: s.dropna().nunique())

    categorical_vars = [col for col in available_vars if df[col].dtype == object or nunique[col] <= 4]
    numeric_vars = [col for col in available_vars if col not in categorical_vars]

    processed_numeric = coerced[numeric_vars].dropna() if numeric_vars else pd.DataFrame()
    statistical_parameters = {}

    for col in numeric_vars:
        data = processed_numeric[col].to_numpy()
        if data.size == 0:
            continue

        mean_val = float(np.mean(data))
        var_val = float(np.var(data, ddof=1))
        skew_val = float(stats.skew(data))
        kurtosis_val = float(stats.kurtosis(data))

        parametric_fits = {}
        for dist_name, dist_func in [("Normal", stats.norm), ("Log-Normal", stats.lognorm), ("Gamma", stats.gamma)]:
            try:
                if dist_name == "Normal":
                    loc, scale = dist_func.fit(data)
                    parametric_fits[dist_name] = {"loc": round(float(loc), 4), "scale": round(float(scale), 4)}
                else:
                    shape, loc, scale = dist_func.fit(data)
                    parametric_fits[dist_name] = {
                        "shape": round(float(shape), 4),
                        "loc": round(float(loc), 4),
                        "scale": round(float(scale), 4),
                    }
            except Exception:
                parametric_fits[dist_name] = None

        statistical_parameters[col] = {
            "is_categorical": False,
            "Moments": {
                "Mean": round(mean_val, 4),
                "Variance": round(var_val, 4),
                "Skewness": round(skew_val, 4),
                "Kurtosis": round(kurtosis_val, 4),
            },
            "Parametric_Fits": parametric_fits,
            "Empirical_Values": data.tolist(),
        }

    for col in categorical_vars:
        series = df[col].dropna().astype(str)
        if series.empty:
            cats = []
            probs = []
        else:
            cats, counts = np.unique(series.values, return_counts=True)
            probs = (counts / counts.sum()).tolist()
            cats = [str(x) for x in cats]

        statistical_parameters[col] = {
            "is_categorical": True,
            "Categories": [{"value": cats[i], "prob": round(probs[i], 6)} for i in range(len(cats))],
            "Empirical_Values": series.tolist(),
        }

    corr_df = df[available_vars].copy()
    for col in numeric_vars:
        corr_df[col] = coerced[col]
    for col in categorical_vars:
        corr_df[col] = pd.Categorical(corr_df[col]).codes
    corr_df = corr_df.dropna()
    if corr_df.shape[0] > 0:
        corr_matrix = corr_df.corr(method="spearman").values
    else:
        corr_matrix = np.eye(len(available_vars))

    statistical_parameters["_Correlation_Matrix"] = corr_matrix
    statistical_parameters["_Correlation_VARS"] = available_vars

    cleaned_df = df[available_vars].copy()
    for col in numeric_vars:
        cleaned_df[col] = coerced[col]
    cleaned_df = cleaned_df.dropna()
    return {
        "status": "success",
        "cleaned_data": cleaned_df,
        "cleaned_real_data": cleaned_df,
        "parameters": statistical_parameters,
        "available_variables": available_vars,
    }


def _resolve_data_file_path(file_name: str = "combined_raw_data.xlsx"):
    candidate_paths = []
    project_root = os.path.dirname(os.path.abspath(__file__))
    search_roots = [
        os.getcwd(),
        project_root,
        os.path.join(project_root, "data5ปี"),
        os.path.join(project_root, "../"),
    ]

    if file_name:
        candidate_paths.append(file_name)
        if file_name.lower().endswith(('.csv', '.xlsx', '.xls')):
            stem = file_name.rsplit('.', 1)[0]
            candidate_paths.extend([f"{stem}.csv", f"{stem}.xlsx", f"{stem}.xls"])

    candidate_paths.extend([
        "combined_raw_data.xlsx",
        "combined_raw_data.csv",
        "combined_raw_data (1).xlsx",
        "combined_raw_data (1).csv",
        "ต้นฉบับ combine raw data.xlsx",
        "ต้นฉบับ combine raw data.csv",
    ])

    unique_candidates = []
    seen = set()
    for p in candidate_paths:
        if p and p not in seen:
            seen.add(p)
            unique_candidates.append(p)

    for root in search_roots:
        if not root:
            continue
        try:
            for entry in os.listdir(root):
                if "combined_raw_data" in entry.lower() and entry.lower().endswith(('.csv', '.xlsx', '.xls')):
                    unique_candidates.append(os.path.join(root, entry))
        except Exception:
            pass

    for path in unique_candidates:
        if not path:
            continue
        if os.path.exists(path):
            return path

    for root in search_roots:
        if not root:
            continue
        for path in [
            os.path.join(root, "combined_raw_data.xlsx"),
            os.path.join(root, "combined_raw_data.csv"),
            os.path.join(root, "combined_raw_data (1).xlsx"),
            os.path.join(root, "combined_raw_data (1).csv"),
        ]:
            if os.path.exists(path):
                return path
    return None


def load_and_extract_base_parameters(file_path: str = "combined_raw_data (1).csv"):
    resolved_path = _resolve_data_file_path(file_path) if file_path else None
    if resolved_path is None:
        return {"status": "error", "message": f"❌ ไม่พบไฟล์ข้อมูลจริง ({file_path}) ในโฟลเดอร์โปรเจค"}

    try:
        df_input = pd.read_csv(resolved_path) if resolved_path.lower().endswith('.csv') else pd.read_excel(resolved_path)
    except FileNotFoundError:
        return {"status": "error", "message": f"❌ ไม่พบไฟล์ข้อมูลจริง ({file_path}) ในโฟลเดอร์โปรเจค"}

    try:
        if 'LDL2' in df_input.columns:
            df_input.rename(columns={'LDL2': 'LDL'}, inplace=True)
    except Exception:
        pass

    return process_and_extract_parameters(df_input)


def _build_latent_copula_cholesky(corr_matrix: np.ndarray, cache: dict | None = None) -> np.ndarray:
    corr = np.asarray(corr_matrix, dtype=np.float32)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    if cache is not None:
        cached = cache.get("_copula_cholesky")
        if cached is not None and np.allclose(cached[0], corr):
            return cached[1]

    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.clip(eigvals, 1e-8, None)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    L = np.asarray(L, dtype=np.float32)
    if cache is not None:
        cache["_copula_cholesky"] = (corr.copy(), L.copy())
    return L


def _kde_inv_cdf_lookup(values: np.ndarray, u: np.ndarray, grid_size: int = 1000) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    u = np.clip(np.asarray(u, dtype=np.float64), 1e-6, 1.0 - 1e-6)

    if data.size == 0:
        return np.zeros_like(u, dtype=np.float64)
    if data.size == 1:
        return np.full_like(u, float(data[0]), dtype=np.float64)

    percentiles = np.clip(u * 100.0, 0.0, 100.0)
    return np.percentile(data, percentiles).astype(np.float64, copy=False)


def _match_output_precision(reference_values: np.ndarray, generated_values: np.ndarray):
    ref = pd.to_numeric(pd.Series(reference_values), errors="coerce").dropna().to_numpy()
    generated_values = np.asarray(generated_values, dtype=float)
    if ref.size == 0:
        return generated_values

    if np.all(np.isclose(ref, np.round(ref), atol=1e-8)):
        return np.rint(generated_values).astype(int)

    max_decimals = 0
    for value in ref:
        if pd.isna(value):
            continue
        text = format(float(value), ".12f").rstrip("0").rstrip(".")
        if "." in text:
            max_decimals = max(max_decimals, len(text.split(".")[-1]))
    if max_decimals > 0:
        return np.round(generated_values, decimals=max_decimals)

    return generated_values


def generate_synthetic_data(parameters: dict, num_samples: int = 1000, selected_variables: list | None = None, include_synthetic_hn: bool = False, random_seed: int | None = None) -> pd.DataFrame:
    """Generate synthetic records with Skewness-driven Parametric Selection and Gaussian Copula."""
    vars_list = [k for k in parameters.keys() if not k.startswith("_")]
    if selected_variables:
        vars_list = [v for v in selected_variables if v in vars_list]
    num_vars = len(vars_list)

    full_vars = [k for k in parameters.keys() if not k.startswith("_")]
    corr_matrix = parameters.get("_Correlation_Matrix")
    corr_vars = parameters.get("_Correlation_VARS", full_vars)
    if corr_matrix is None:
        corr_matrix = np.eye(num_vars)
    else:
        corr_matrix = np.array(corr_matrix, dtype=float)
        if len(corr_vars) == corr_matrix.shape[0]:
            idxs = [corr_vars.index(v) for v in vars_list]
            corr_matrix = corr_matrix[np.ix_(idxs, idxs)]

    if corr_matrix.size == 0:
        corr_matrix = np.eye(num_vars)

    spearman = np.array(corr_matrix, dtype=float)
    pearson_latent = 2.0 * np.sin((np.pi / 6.0) * spearman)
    pearson_latent = (pearson_latent + pearson_latent.T) / 2.0
    np.fill_diagonal(pearson_latent, 1.0)

    try:
        from statsmodels.stats.correlation_tools import corr_nearest
        pearson_pd = corr_nearest(pearson_latent)
    except ModuleNotFoundError:
        raise ModuleNotFoundError("statsmodels is required for corr_nearest. Please install it: pip install statsmodels")
    except Exception:
        pearson_pd = pearson_latent

    pearson_pd = (pearson_pd + pearson_pd.T) / 2.0
    np.fill_diagonal(pearson_pd, 1.0)

    rng = np.random.default_rng(random_seed)
    copula_cache = parameters.setdefault("_copula_cache", {})
    L = _build_latent_copula_cholesky(pearson_pd, cache=copula_cache)
    Z = rng.standard_normal(size=(num_samples, num_vars)).astype(np.float32, copy=False)
    Z = (Z @ L.T).astype(np.float32, copy=False)
    U = stats.norm.cdf(Z).astype(np.float32, copy=False)

    out = pd.DataFrame(index=range(num_samples), columns=vars_list)

    for j, col in enumerate(vars_list):
        col_params = parameters[col]
        if col_params.get("is_categorical"):
            cats = [c["value"] for c in col_params.get("Categories", [])]
            probs = np.array([c["prob"] for c in col_params.get("Categories", [])], dtype=np.float64)
            if probs.sum() <= 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / probs.sum()
            cum = np.cumsum(probs)
            u = U[:, j]
            idxs = np.searchsorted(cum, u, side="right")
            idxs = np.clip(idxs, 0, len(cats) - 1)
            out[col] = np.array([cats[i] for i in idxs], dtype=object)
            continue

        raw_vals = col_params.get("Empirical_Values", [])
        empirical_data = pd.to_numeric(pd.Series(raw_vals), errors="coerce").dropna().to_numpy(dtype=np.float64)
        if empirical_data.size == 0:
            out[col] = np.nan
            continue

        u = U[:, j]
        u_clipped = np.clip(u, 1e-6, 1.0 - 1e-6)
        mapped = None

        moments = col_params.get("Moments", {})
        skewness = moments.get("Skewness", 0.0)
        param_fits = col_params.get("Parametric_Fits", {})

        # 1. กลุ่มความซับซ้อนสูง (Lipids, Bimodal Glucose, Heavy-tail AST, Discrete Age) -> PCHIP
        high_complexity_vars = {"CHO", "TG", "LDL", "GLUCOSE", "GLU", "AST", "AGE"}
        
        if str(col).upper() in high_complexity_vars and empirical_data.size >= 3:
            try:
                from scipy.interpolate import PchipInterpolator
                grid_size = 300
                q_grid = np.linspace(0.0, 1.0, grid_size)
                x_vals = np.quantile(empirical_data, q_grid, method="linear")
                pchip = PchipInterpolator(q_grid, x_vals)
                mapped = pchip(u_clipped)
                mapped = np.clip(mapped, np.min(empirical_data), np.max(empirical_data))
                
                if str(col).upper() == "AGE":
                    mapped = np.round(mapped)
                    
                print(f"[Step 2] {col}: Fitted with High-Fidelity PCHIP Quantile Mapping")
            except Exception:
                mapped = None

        # 2. ตัวแปรอื่นๆ: ใช้ Parametric Skewness Gates ตามระเบียบวิธีวิจัย
        if mapped is None:
            if skewness > 0.8 and param_fits.get("Log-Normal"):
                try:
                    fit = param_fits["Log-Normal"]
                    mapped = stats.lognorm.ppf(u_clipped, s=fit["shape"], loc=fit["loc"], scale=fit["scale"])
                    print(f"[Step 2] {col} (Skew={skewness:.2f}): Matched with Log-Normal Distribution")
                except Exception:
                    mapped = None

            elif skewness > 0.4 and param_fits.get("Gamma"):
                try:
                    fit = param_fits["Gamma"]
                    mapped = stats.gamma.ppf(u_clipped, a=fit["shape"], loc=fit["loc"], scale=fit["scale"])
                    print(f"[Step 2] {col} (Skew={skewness:.2f}): Matched with Gamma Distribution")
                except Exception:
                    mapped = None

            elif param_fits.get("Normal"):
                try:
                    fit = param_fits["Normal"]
                    mapped = stats.norm.ppf(u_clipped, loc=fit["loc"], scale=fit["scale"])
                    print(f"[Step 2] {col} (Skew={skewness:.2f}): Matched with Normal Distribution")
                except Exception:
                    mapped = None

        # 3. Fallback เมื่อฟิตสมการไม่ผ่าน
        if mapped is None or np.any(np.isnan(mapped)) or np.any(np.isinf(mapped)):
            mapped = _kde_inv_cdf_lookup(empirical_data, u, grid_size=1000)

        mapped = np.asarray(mapped, dtype=np.float64)
        if np.nanmin(empirical_data) >= 0:
            mapped = np.where(mapped < 0.0, 0.0, mapped)

        mapped = _match_output_precision(empirical_data, mapped)
        out[col] = mapped

    # Biological consistency: LDL <= CHO - HDLC
    try:
        if "LDL" in out.columns and "CHO" in out.columns and "HDLC" in out.columns:
            cho = pd.to_numeric(out["CHO"], errors="coerce").to_numpy(dtype=np.float64)
            hdlc = pd.to_numeric(out["HDLC"], errors="coerce").to_numpy(dtype=np.float64)
            ldl = pd.to_numeric(out["LDL"], errors="coerce").to_numpy(dtype=np.float64)
            allowed = cho - hdlc
            mask = (np.isfinite(allowed) & np.isfinite(ldl)) & (ldl > allowed)
            if np.any(mask):
                adjusted = np.maximum(0.0, np.minimum(ldl[mask], allowed[mask] - 0.5))
                out.loc[mask, "LDL"] = adjusted
    except Exception:
        pass

    return out


def _moment_error_score(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    common = [c for c in df_real.columns if c in df_synth.columns]
    num_cols = [c for c in common if pd.api.types.is_numeric_dtype(df_real[c]) and pd.api.types.is_numeric_dtype(df_synth[c])]
    if not num_cols:
        return float('inf')

    diffs = []
    for col in num_cols:
        r = pd.to_numeric(df_real[col].dropna(), errors='coerce').dropna()
        s = pd.to_numeric(df_synth[col].dropna(), errors='coerce').dropna()
        if len(r) < 2 or len(s) < 2:
            continue
        sk_r = float(stats.skew(r, bias=False))
        sk_s = float(stats.skew(s, bias=False))
        kt_r = float(stats.kurtosis(r, fisher=False, bias=False))
        kt_s = float(stats.kurtosis(s, fisher=False, bias=False))
        diffs.append(abs(sk_r - sk_s))
        diffs.append(abs(kt_r - kt_s))

    if not diffs:
        return float('inf')
    return float(np.nanmean(diffs))


def generate_with_seed_search(parameters: dict, df_real: pd.DataFrame, num_samples: int = 1000, num_trials: int = 30, selected_variables: list | None = None) -> tuple:
    best_score = float('inf')
    best_seed = None
    best_synth = None

    for seed in range(num_trials):
        try:
            synth = generate_synthetic_data(parameters, num_samples=num_samples, selected_variables=selected_variables, random_seed=seed)
        except Exception:
            continue

        score = _moment_error_score(df_real, synth)
        if score < best_score:
            best_score = score
            best_seed = seed
            best_synth = synth

    if best_seed is not None:
        final_synth = generate_synthetic_data(parameters, num_samples=num_samples, selected_variables=selected_variables, random_seed=best_seed)
        return final_synth, best_seed, best_score
    else:
        synth = generate_synthetic_data(parameters, num_samples=num_samples, selected_variables=selected_variables, random_seed=None)
        return synth, None, float('inf')
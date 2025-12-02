#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:45:42 2025

@author: sebastianlouis
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from scipy import stats

st.set_page_config(page_title="Histogram Fitter", layout="wide")

st.title("Histogram Fitter with SciPy Distributions")
st.write(
    "Upload or enter data, fit multiple probability distributions, "
    "and explore manual parameter controls."
)



def parse_text_data(text: str) -> np.ndarray:
    text = text.replace("\n", ",").replace(";", ",")
    arr = np.fromstring(text, sep=",")
    return arr[~np.isnan(arr)]

def load_csv_file(uploaded_file) -> np.ndarray:
    try:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return np.array([])
        col = st.selectbox(
            "Select column from uploaded CSV",
            list(numeric_cols),
            key="csv_column_select",
        )
        return df[col].dropna().values
    except Exception:
        return np.array([])

def compute_errors_from_pdf(data: np.ndarray, dist_obj, bins: int = 30):
    """
    Approximate error between empirical histogram and distribution PDF.
    """
    hist_vals, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    pdf_vals = dist_obj.pdf(bin_centers)
    errors = np.abs(hist_vals - pdf_vals)
    return errors, bin_centers, hist_vals, pdf_vals

DIST_OPTIONS = {
    "Normal (norm)": stats.norm,
    "Gamma (gamma)": stats.gamma,
    "Exponential (expon)": stats.expon,
    "Lognormal (lognorm)": stats.lognorm,
    "Weibull (weibull_min)": stats.weibull_min,
    "Beta (beta)": stats.beta,
    "Uniform (uniform)": stats.uniform,
    "Chi-squared (chi2)": stats.chi2,
    "Student-t (t)": stats.t,
    "Cauchy (cauchy)": stats.cauchy,
}


left_col, right_col = st.columns([1.1, 1.9])

with left_col:
    st.subheader("1. Input Data")

    data_source = st.radio(
        "Data source",
        ["Manual entry", "Upload CSV"],
        horizontal=True,
    )

    data = np.array([])
    if data_source == "Manual entry":
        example = "1.1, 2.3, 2.5, 3.0, 4.7"
        manual_text = st.text_area(
            "Enter numbers (comma, space, or newline separated):",
            value="",
            placeholder=example,
            height=150,
        )
        if manual_text.strip():
            data = parse_text_data(manual_text)
    else:
        uploaded_file = st.file_uploader(
            "Upload a CSV file (first numeric column will be used)",
            type=["csv"],
        )
        if uploaded_file is not None:
            data = load_csv_file(uploaded_file)

    if data.size == 0:
        st.info("Enter or upload numeric data to begin fitting.")
    else:
        st.success(f"Loaded {data.size} data points.")

    st.subheader("2. Distribution options")
    chosen_dists = st.multiselect(
        "Select distributions to fit (at least one):",
        list(DIST_OPTIONS.keys()),
        default=[
            "Normal (norm)",
            "Gamma (gamma)",
            "Exponential (expon)",
        ],
    )

    bins = st.slider("Number of histogram bins", 10, 80, 30)

    st.subheader("3. Manual fitting controls")
    st.write(
        "Choose one distribution below for interactive, manual parameter control."
    )
    manual_dist_name = st.selectbox(
        "Manual-fit distribution",
        list(DIST_OPTIONS.keys()),
        index=0,
    )

with right_col:
    st.subheader("4. Results and visualization")

    if data.size == 0 or len(chosen_dists) == 0:
        st.write("Waiting for data and distribution selections…")
    else:
        fit_results = {}
        for name in chosen_dists:
            dist = DIST_OPTIONS[name]
            try:
                params = dist.fit(data)
                dist_obj = dist(*params)
                errors, _, _, _ = compute_errors_from_pdf(data, dist_obj, bins=bins)
                avg_err = float(np.mean(errors))
                max_err = float(np.max(errors))
                fit_results[name] = {
                    "dist": dist,
                    "params": params,
                    "avg_err": avg_err,
                    "max_err": max_err,
                    "dist_obj": dist_obj,
                }
            except Exception as e:
                fit_results[name] = {"error": str(e)}

        valid_fits = {
            k: v for k, v in fit_results.items() if "avg_err" in v
        }
        best_name = None
        if len(valid_fits) > 0:
            best_name = min(valid_fits, key=lambda k: valid_fits[k]["avg_err"])

        tab_auto, tab_manual, tab_table = st.tabs(
            ["Auto fit & plot", "Manual fit sliders", "Fit statistics"]
        )

        with tab_auto:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(data, bins=bins, density=True, alpha=0.4, label="Data")

            x_vals = np.linspace(data.min(), data.max(), 400)

            for name, res in fit_results.items():
                if "dist_obj" not in res:
                    continue
                pdf_vals = res["dist_obj"].pdf(x_vals)
                label = f"{name} (avg err={res['avg_err']:.3g})"
                if name == best_name:
                    label += "  ← best"
                ax.plot(x_vals, pdf_vals, label=label)

            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
            ax.set_title("Histogram and fitted PDFs")

            st.pyplot(fig)

        with tab_manual:
            dist = DIST_OPTIONS[manual_dist_name]

            st.markdown("**Auto-fit parameters (starting point):**")
            auto_params = None
            if manual_dist_name in fit_results and "params" in fit_results[manual_dist_name]:
                auto_params = fit_results[manual_dist_name]["params"]
                st.write(tuple(float(p) for p in auto_params))
            else:
                st.write("Auto-fit failed; sliders will use default guesses.")
                auto_params = (1.0, 0.0, 1.0)

            st.markdown("**Adjust parameters:**")

            slider_values = []
            for i, p in enumerate(auto_params[:4]):
                center = float(p)
                span = max(abs(center), 1.0)
                min_val = center - 2 * span
                max_val = center + 2 * span
                slider = st.slider(
                    f"Param {i+1}",
                    float(min_val),
                    float(max_val),
                    float(center),
                    key=f"slider_{manual_dist_name}_{i}",
                )
                slider_values.append(slider)

            slider_params = tuple(slider_values[: len(auto_params)])

            try:
                manual_dist_obj = dist(*slider_params)
                errors, bin_centers, hist_vals, pdf_vals = compute_errors_from_pdf(
                    data, manual_dist_obj, bins=bins
                )
                avg_err = float(np.mean(errors))
                max_err = float(np.max(errors))

                st.write(
                    f"**Manual parameters:** {tuple(round(v, 4) for v in slider_params)}"
                )
                st.write(
                    f"Average error (hist vs PDF): `{avg_err:.4g}`, "
                    f"Max error: `{max_err:.4g}`"
                )

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.hist(data, bins=bins, density=True, alpha=0.4, label="Data")
                x_vals = np.linspace(data.min(), data.max(), 400)
                ax2.plot(x_vals, manual_dist_obj.pdf(x_vals), "r-", label="Manual fit")
                ax2.set_xlabel("Value")
                ax2.set_ylabel("Density")
                ax2.legend()
                ax2.set_title(f"Manual fit: {manual_dist_name}")
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Could not create manual distribution: {e}")

        with tab_table:
            rows = []
            for name, res in fit_results.items():
                if "error" in res:
                    rows.append(
                        {
                            "Distribution": name,
                            "Status": "Fit failed",
                            "Parameters": "",
                            "Average error": np.nan,
                            "Max error": np.nan,
                        }
                    )
                    continue
                rows.append(
                    {
                        "Distribution": name,
                        "Status": "OK" + (" (best)" if name == best_name else ""),
                        "Parameters": tuple(float(p) for p in res["params"]),
                        "Average error": res["avg_err"],
                        "Max error": res["max_err"],
                    }
                )
            if rows:
                stats_df = pd.DataFrame(rows)
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.write("No successful fits to summarize.")

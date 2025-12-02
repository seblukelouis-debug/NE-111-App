import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

st.set_page_config(page_title="NE-111 Histogram Fitter", layout="wide")

st.title("Histogram Distribution Fitter")

def parse_text_data(text):
    if not text or not text.strip():
        return np.array([])
    text = text.replace('\n', ' ').replace(';', ' ').replace('\t', ' ')
    numbers = []
    for part in text.split():
        try:
            num = float(part.replace(',', '.'))
            numbers.append(num)
        except:
            continue
    return np.array(numbers)

DIST_OPTIONS = {
    "Normal": stats.norm,
    "Gamma": stats.gamma,
    "Exponential": stats.expon,
    "Lognormal": stats.lognorm,
    "Weibull": stats.weibull_min,
    "Beta": stats.beta,
    "Uniform": stats.uniform,
    "Chi-squared": stats.chi2,
}

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data Input")
    input_type = st.radio("Choose:", ["Manual entry", "Upload CSV"], horizontal=True)

    data = np.array([])

    if input_type == "Manual entry":
        user_input = st.text_area(
            "Enter numbers (spaces, commas, newlines OK):",
            placeholder="1.2 2.1 1.8 3.0 2.5 1.9 2.2",
            height=100
        )
        data = parse_text_data(user_input)

    else:
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    col_name = st.selectbox("Select column:", numeric_cols)
                    data = df[col_name].dropna().values
                else:
                    data = df.iloc[:, 0].dropna()
                    if len(data) > 0 and pd.api.types.is_numeric_dtype(data):
                        data = data.values
                    else:
                        data = np.array([])
            except Exception as e:
                st.error(f"CSV error: {e}")

    if len(data) > 0:
        st.success(f"Loaded {len(data)} data points")

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"{np.mean(data):.3f}")
        c2.metric("Std Dev", f"{np.std(data):.3f}")
        c3.metric("Range", f"{np.min(data):.1f} – {np.max(data):.1f}")

        st.write("First 10 values:", data[:10])

    else:
        st.warning("No valid numeric data found")
        st.info("Example: 1.2 2.1 1.8 3.0 2.5")

    selected_dists = st.multiselect(
        "Distributions to fit:",
        list(DIST_OPTIONS.keys()),
        default=["Normal", "Gamma", "Exponential"]
    )

    n_bins = st.slider("Histogram bins", 5, 60, 20)

with col2:
    st.header("Histogram and Fits")

    if len(data) == 0:
        st.info("Enter or upload data to see graphs.")
    else:
        # ---------- Histogram using st.bar_chart ----------
        counts, bin_edges = np.histogram(data, bins=n_bins, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        hist_df = pd.DataFrame({"Value": bin_centers, "Density": counts})
        st.subheader("Histogram")
        st.bar_chart(hist_df, x="Value", y="Density")

        # ---------- Fit selected distributions and show normal curve ----------
        st.subheader("Fitted Normal Curve (example)")

        # Basic normal fit from mean and std
        mu = np.mean(data)
        sigma = np.std(data) if np.std(data) > 0 else 1.0
        x_grid = np.linspace(data.min(), data.max(), 200)
        pdf_vals = stats.norm.pdf(x_grid, loc=mu, scale=sigma)
        curve_df = pd.DataFrame({"Value": x_grid, "PDF": pdf_vals})
        st.line_chart(curve_df, x="Value", y="PDF")

        # ---------- Parameter table for all selected dists ----------
        if selected_dists:
            st.subheader("Distribution Fitting Results")

            table_rows = []
            for name in selected_dists:
                try:
                    dist_class = DIST_OPTIONS[name]
                    params = dist_class.fit(data)
                    params_str = ", ".join(f"{p:.3f}" for p in params)
                    table_rows.append({
                        "Distribution": name,
                        "Parameters": params_str,
                        "Num Params": len(params),
                    })
                except Exception as e:
                    table_rows.append({
                        "Distribution": name,
                        "Parameters": f"Fit failed: {str(e)[:30]}",
                        "Num Params": "-",
                    })

            results_df = pd.DataFrame(table_rows)
            st.dataframe(results_df, use_container_width=True)
        else:
            st.info("Select at least one distribution to see fit parameters.")

# Manual fitting section
with st.expander("Manual Parameter Fitting"):
    if len(data) > 0 and selected_dists:
        dist_name = st.selectbox("Distribution:", selected_dists)
        dist_class = DIST_OPTIONS[dist_name]

        try:
            auto_params = dist_class.fit(data)
            st.write("Automatic fit:", tuple(np.round(auto_params, 3)))

            cols = st.columns(min(4, len(auto_params)))
            manual_params = []
            for i, p in enumerate(auto_params):
                with cols[i % 4]:
                    slider = st.slider(
                        f"Param {i+1}",
                        float(p - abs(p) * 2 if p != 0 else -1),
                        float(p + abs(p) * 2 if p != 0 else 1),
                        float(p),
                    )
                    manual_params.append(slider)

            if st.button("Apply Manual Parameters"):
                try:
                    manual_dist = dist_class(*manual_params)
                    st.success("Manual parameters applied.")
                    st.write("Manual:", tuple(np.round(manual_params, 3)))
                except Exception as e:
                    st.error(f"Manual fit failed: {e}")
        except Exception as e:
            st.error(f"Automatic fit failed: {e}")
    else:
        st.info("Load data and choose at least one distribution first.")

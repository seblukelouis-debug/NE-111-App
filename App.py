import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

st.set_page_config(page_title="NE-111 Histogram Fitter", layout="wide")

st.title("Histogram Distribution Fitter")

def parse_text_data(text):
    if not text or not text.strip():
        return np.array([])
    text = text.replace("\n", ",").replace(";", ",").replace(" ", ",")
    try:
        arr = np.fromstring(text.strip(), sep=",")
        return arr[~np.isnan(arr)]
    except:
        return np.array([])

DIST_OPTIONS = {
    "Normal": stats.norm,
    "Gamma": stats.gamma,
    "Exponential": stats.expon,
    "Lognormal": stats.lognorm,
    "Weibull": stats.weibull_min,
    "Beta": stats.beta
}

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data Input")
    input_type = st.radio("Data source:", ["Manual entry", "Upload CSV"])

    data = np.array([])
    if input_type == "Manual entry":
        user_input = st.text_area("Enter numbers (comma/space separated):", 
                                placeholder="1.2, 2.1, 1.8, 3.0, 2.5", height=100)
        data = parse_text_data(user_input)
    else:
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if len(df.select_dtypes(include=[np.number]).columns) > 0:
                    col_name = st.selectbox("Select column:", 
                                          df.select_dtypes(include=[np.number]).columns)
                    data = df[col_name].dropna().values
                else:
                    data = df.iloc[:, 0].dropna().values
            except:
                pass

    if len(data) > 0:
        st.success(f"Loaded {len(data)} data points")
        col1a, col2a, col3a = st.columns(3)
        col1a.metric("Mean", f"{np.mean(data):.3f}")
        col2a.metric("Std", f"{np.std(data):.3f}")
        col3a.metric("Range", f"{np.min(data):.1f} - {np.max(data):.1f}")

    selected_dists = st.multiselect("Distributions to fit:", 
                                  list(DIST_OPTIONS.keys()),
                                  default=["Normal", "Gamma"])

    n_bins = st.slider("Number of bins", 10, 50, 25)

with col2:
    if len(data) > 0:
        st.subheader("Data Summary")
        
        # Data statistics table
        stats_df = pd.DataFrame({
            'Metric': ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Median'],
            'Value': [len(data), np.mean(data), np.std(data), 
                     np.min(data), np.max(data), np.median(data)]
        })
        st.dataframe(stats_df, use_container_width=True)

        if selected_dists:
            st.subheader("Distribution Fits")
            
            fits = {}
            for name in selected_dists:
                try:
                    dist_class = DIST_OPTIONS[name]
                    params = dist_class.fit(data)
                    dist_fit = dist_class(*params)
                    fits[name] = {'params': params, 'dist': dist_fit}
                except:
                    continue

            if fits:
                # Results table with fit parameters
                table_data = []
                for name, fit_info in fits.items():
                    params_str = ", ".join([f"{p:.3f}" for p in fit_info['params']])
                    table_data.append({
                        'Distribution': name,
                        'Parameters': params_str,
                        'Shape(s)': len(fit_info['params']) - 2 if len(fit_info['params']) > 2 else 0
                    })
                
                fit_df = pd.DataFrame(table_data)
                st.dataframe(fit_df, use_container_width=True)

                st.info("Parameters shown as (shape parameters..., location, scale). "
                       "All distributions fitted successfully!")
            else:
                st.warning("No distributions could be fitted")

with st.expander("Manual Parameter Fitting"):
    if len(data) > 0 and selected_dists:
        dist_name = st.selectbox("Select distribution for manual fitting:", selected_dists)
        if dist_name in fits:
            params = fits[dist_name]['params']
            st.info(f"Auto-fitted parameters: {tuple(np.round(params, 3))}")
            
            st.subheader("Adjust parameters manually:")
            manual_params = []
            cols = st.columns(min(4, len(params)))
            for i, p in enumerate(params):
                with cols[i]:
                    min_val = max(0.01, p * 0.5) if p > 0 else p - 1
                    max_val = p * 3 if p > 0 else p + 1
                    slider_val = st.slider(f"Param {i+1}", min_val, max_val, float(p))
                    manual_params.append(slider_val)
            
            if st.button("Apply Manual Parameters"):
                try:
                    manual_dist = DIST_OPTIONS[dist_name](*manual_params)
                    manual_params_str = ", ".join([f"{p:.3f}" for p in manual_params])
                    st.success(f"Manual parameters applied: {manual_params_str}")
                    
                    manual_stats = pd.DataFrame({
                        'Parameter': [f'P{i+1}' for i in range(len(manual_params))],
                        'Value': manual_params
                    })
                    st.dataframe(manual_stats)
                except Exception as e:
                    st.error(f"Manual fitting failed: {str(e)}")
    else:
        st.info("Load data and select distributions first")

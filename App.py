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
    "Chi-squared": stats.chi2
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
                st.write("CSV columns:", list(df.columns))
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    col_name = st.selectbox("Select column:", numeric_cols)
                    data = df[col_name].dropna().values
                else:
                    data = df.iloc[:, 0].dropna()
                    if len(data) > 0 and pd.api.types.is_numeric_dtype(data):
                        data = data.values
            except Exception as e:
                st.error(f"CSV error: {e}")

    if len(data) > 0:
        st.success(f"Loaded {len(data)} data points")
        
        col1m, col2m, col3m = st.columns(3)
        col1m.metric("Mean", f"{np.mean(data):.3f}")
        col2m.metric("Std Dev", f"{np.std(data):.3f}")
        col3m.metric("Range", f"{np.min(data):.1f} - {np.max(data):.1f}")
        
        st.write("First 10 values:", data[:10])
        
    else:
        st.warning("No valid numeric data found")
        st.info("Example: 1.2 2.1 1.8 3.0 2.5")

    selected_dists = st.multiselect(
        "Distributions to fit:", 
        list(DIST_OPTIONS.keys()),
        default=["Normal", "Gamma", "Exponential"]
    )

with col2:
    if len(data) > 0 and len(selected_dists) > 0:
        st.subheader("Distribution Fitting Results")
        
        fits = {}
        for name in selected_dists:
            try:
                dist_class = DIST_OPTIONS[name]
                params = dist_class.fit(data)
                dist_fit = dist_class(*params)
                fits[name] = {'params': params, 'dist': dist_fit}
            except Exception as e:
                st.warning(f"{name} fit failed")

        if fits:
            table_data = []
            for name, fit_info in fits.items():
                params_str = ", ".join([f"{p:.3f}" for p in fit_info['params']])
                table_data.append({
                    'Distribution': name,
                    'Parameters': params_str,
                    'Num Params': len(fit_info['params'])
                })
            
            df_results = pd.DataFrame(table_data)
            st.dataframe(df_results, use_container_width=True)
            
            st.caption("Parameters: (shape parameter(s), location, scale)")

with st.expander("Manual Parameter Fitting"):
    if len(data) > 0 and 'fits' in locals() and fits:
        dist_name = st.selectbox("Distribution:", list(fits.keys()))
        params = fits[dist_name]['params']
        
        st.info(f"Automatic fit: {tuple(np.round(params, 3))}")
        
        col_params = st.columns(3)
        manual_params = []
        for i, p in enumerate(params):
            with col_params[i % 3]:
                slider = st.slider(f"P{i+1}", 0.0, max(1.0, float(p*2)), float(p))
                manual_params.append(slider)
        
        if st.button("Apply Manual Fit"):
            try:
                manual_dist = DIST_OPTIONS[dist_name](*manual_params)
                st.success("Manual fit applied!")
                st.code(f"Manual parameters: {tuple(np.round(manual_params, 3))}")
            except:
                st.error("Invalid manual parameters")
    else:
        st.info("Load data and fit distributions first")

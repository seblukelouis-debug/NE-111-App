import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

st.set_page_config(page_title="NE-111 Histogram Fitter", layout="wide")

st.title("Histogram Distribution Fitter")

def parse_text_data(text):
    """Robust parser - handles ANY input format"""
    if not text or not text.strip():
        return np.array([])
    
    # Replace all possible separators
    text = text.replace('\n', ' ').replace(';', ' ').replace('\t', ' ')
    text = ''.join(c for c in text if c.isdigit() or c == '.' or c == '-' or c.isspace())
    
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
    
    # Show raw data preview first
    st.subheader("Step 1: Enter Data")
    input_type = st.radio("Choose:", ["Manual entry", "Upload CSV"], horizontal=True)
    
    data = np.array([])
    
    if input_type == "Manual entry":
        user_input = st.text_area(
            "Paste numbers here (any format works):", 
            placeholder="1.2 2.1 1.8 3.0 2.5 1.9 2.2\nOR\n1,2,1.8,3,2.5", 
            height=120
        )
        data = parse_text_data(user_input)
        
    else:
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("CSV preview:")
                st.dataframe(df.head())
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    col_name = st.selectbox("Select column:", numeric_cols)
                    data = df[col_name].dropna().values
                else:
                    st.warning("No numeric columns found")
                    data = df.iloc[:, 0].dropna()
                    if pd.api.types.is_numeric_dtype(data):
                        data = data.values
                    else:
                        data = np.array([])
            except Exception as e:
                st.error(f"CSV error: {e}")
    
    # Data validation and display
    if len(data) > 0:
        st.success(f"✅ SUCCESS! Loaded {len(data)} data points")
        
        col1m, col2m, col3m = st.columns(3)
        col1m.metric("Mean", f"{np.mean(data):.3f}")
        col2m.metric("Std Dev", f"{np.std(data):.3f}")
        col3m.metric("Range", f"{np.min(data):.1f} - {np.max(data):.1f}")
        
        st.subheader("Your data:")
        st.write(data[:10], help=f"First 10 of {len(data)} points")
        
    else:
        st.error("❌ No valid data loaded")
        st.info("Try: 1.2, 2.1, 1.8, 3.0, 2.5")

    st.subheader("Step 2: Choose Distributions")
    selected_dists = st.multiselect(
        "Select distributions:", 
        list(DIST_OPTIONS.keys()),
        default=["Normal", "Gamma", "Exponential"]
    )

with col2:
    if len(data) > 0 and selected_dists:
        st.subheader("Fitting Results")
        
        fits = {}
        for name in selected_dists:
            try:
                dist_class = DIST_OPTIONS[name]
                with st.spinner(f"Fitting {name}..."):
                    params = dist_class.fit(data)
                    dist_fit = dist_class(*params)
                    fits[name] = {'params': params, 'dist': dist_fit}
                st.success(f"✅ {name} fitted")
            except Exception as e:
                st.warning(f"⚠️ {name} failed: {str(e)[:50]}")
        
        if fits:
            # Parameters table
            table_data = []
            for name, fit_info in fits.items():
                params_str = ", ".join([f"{p:.4f}" for p in fit_info['params']])
                table_data.append({
                    'Distribution': name,
                    'Parameters': params_str,
                    'Num params': len(fit_info['params'])
                })
            
            st.subheader("Fit Parameters")
            df_fits = pd.DataFrame(table_data)
            st.dataframe(df_fits, use_container_width=True)
            
            st.info("Parameters format: (shape(s)..., location, scale)")
        
        else:
            st.warning("No successful fits")
    elif len(data) == 0:
        st.info("Enter data first")
    else:
        st.info("Select at least one distribution")

# Manual fitting
with st.expander("Advanced: Manual Parameter Control"):
    if len(data) > 0 and fits:
        dist_name = st.selectbox("Manual fit distribution:", list(fits.keys()))
        params = fits[dist_name]['params']
        
        st.info(f"Auto-fitted: {tuple(np.round(params, 3))}")
        
        manual_params = []
        for i, p in enumerate(params):
            manual_params.append(
                st.slider(f"Param {i+1}", 0.0, max(2.0, abs(p)*2), float(p))
            )
        
        if st.button("Apply Manual Parameters"):
            try:
                manual_dist = DIST_OPTIONS[dist_name](*manual_params)
                st.success("Manual parameters applied!")
                st.code(f"Manual: {tuple(np.round(manual_params, 3))}")
            except:
                st.error("Invalid parameters")

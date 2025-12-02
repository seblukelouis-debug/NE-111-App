import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="NE-111 Histogram Fitter", layout="wide")

st.title("📊 Histogram Distribution Fitter")

def parse_text_data(text):
    """FIXED: Properly parse ALL numbers from text input"""
    if not text or not text.strip():
        return np.array([])
    
    
    text = text.replace('\n', ' ').replace(',', ' ').replace(';', ' ').replace('\t', ' ')
    text = ' '.join(text.split()) 
    
    numbers = []
    for part in text.split():
        try:
            num = float(part)
            numbers.append(num)
        except ValueError:
            continue
    
    data_array = np.array(numbers)
    return data_array

DIST_OPTIONS = {
    "Normal": stats.norm,
    "Gamma": stats.gamma, 
    "Exponential": stats.expon,
    "Lognormal": stats.lognorm,
    "Weibull (min)": stats.weibull_min,
    "Beta": stats.beta,
    "Uniform": stats.uniform,
    "Chi-squared": stats.chi2,
    "Student's t": stats.t,
    "Pareto": stats.pareto,
    "Rayleigh": stats.rayleigh,
    "Logistic": stats.logistic
}

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📈 Data Input")
    input_type = st.radio("Choose input:", ["Manual entry", "Upload CSV"], horizontal=True)

    data = np.array([])

    if input_type == "Manual entry":
        user_input = st.text_area(
            "Enter numbers (spaces, newlines, commas OK):",
            placeholder="1.2 2.1 1.8 3.0 2.5 1.9 2.2 4.5 0.9 3.8",
            height=100
        )
        data = parse_text_data(user_input)
        st.write(f"**DEBUG: Parsed {len(data)} numbers**")  
        if len(data) > 0:
            st.write(f"**Sample:** {data[:5]}...")  
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
                    data = df.iloc[:, 0].dropna().values
            except Exception as e:
                st.error(f"CSV error: {e}")

    if len(data) > 0:
        st.success(f"✅ Loaded {len(data)} data points")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"{np.mean(data):.3f}")
        c2.metric("Std Dev", f"{np.std(data):.3f}")
        c3.metric("Range", f"{np.min(data):.1f} – {np.max(data):.1f}")
    else:
        st.warning(" No valid numeric data found")

    selected_dists = st.multiselect(
        "Distributions to fit:",
        list(DIST_OPTIONS.keys()),
        default=["Normal", "Gamma", "Exponential"]
    )
    n_bins = st.slider("Histogram bins", 5, 60, 20)

with col2:
    st.header("Visualization & Results")
    
    if len(data) > 0 and len(selected_dists) > 0:
        # SINGLE PLOT with histogram + ALL fitted curves
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Histogram
        ax.hist(data, bins=n_bins, density=True, alpha=0.6, color='skyblue', 
                label=f'Data (n={len(data)})', edgecolor='black')
        
        # Fit all distributions
        fit_results = {}
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_dists)))
        color_idx = 0
        
        for name in selected_dists:
            try:
                dist_class = DIST_OPTIONS[name]
                params = dist_class.fit(data)
                dist = dist_class(*params)
                x = np.linspace(data.min()*0.9, data.max()*1.1, 200)
                pdf = dist.pdf(x)
                
                ax.plot(x, pdf, linewidth=2.5, color=colors[color_idx], 
                       label=f"{name}", alpha=0.9)
                
                fit_results[name] = {'params': params, 'dist': dist}
                color_idx += 1
            except Exception as e:
                st.error(f"Failed to fit {name}: {str(e)[:50]}")
        
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Histogram with Fitted Distributions', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Fit quality metrics
        st.subheader("📋 Fit Quality Metrics")
        table_rows = []
        counts, bin_edges = np.histogram(data, bins=n_bins, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        for name, result in fit_results.items():
            dist = result['dist']
            params = result['params']
            fit_vals = dist.pdf(bin_centers)
            avg_error = np.mean(np.abs(counts - fit_vals))
            max_error = np.max(np.abs(counts - fit_vals))
            
            table_rows.append({
                "Distribution": name,
                "Parameters": ", ".join(f"{p:.3f}" for p in params),
                "Avg Error": f"{avg_error:.4f}",
                "Max Error": f"{max_error:.4f}"
            })
        
        results_df = pd.DataFrame(table_rows)
        st.dataframe(results_df, use_container_width=True)
        
    elif len(data) == 0:
        st.info( "Enter or upload data first")
    else:
        st.info(" Select at least one distribution")

# Manual fitting
with st.expander("🔧 Manual Parameter Fitting", expanded=False):
    if len(data) > 0 and selected_dists:
        dist_name = st.selectbox("Distribution:", selected_dists)
        dist_class = DIST_OPTIONS[dist_name]
        n_bins = st.slider("Bins for manual plot:", 5, 60, 20)
        
        try:
            auto_params = dist_class.fit(data)
            st.write(f"**Automatic fit:** {tuple(np.round(auto_params, 3))}")
            
            cols = st.columns(min(4, len(auto_params)))
            manual_params = [auto_params[i] for i in range(len(auto_params))]
            
            for i, p in enumerate(auto_params):
                with cols[i % 4]:
                    min_val = float(p - abs(p)*2) if p != 0 else -2.0
                    max_val = float(p + abs(p)*2) if p != 0 else 2.0
                    manual_params[i] = st.slider(
                        f"Param {i+1}", min_val, max_val, float(p), 0.01
                    )
            
            if st.button("🎛️ Apply Manual Parameters"):
                manual_dist = dist_class(*manual_params)
                fig_manual, ax_manual = plt.subplots(figsize=(12, 7))
                ax_manual.hist(data, bins=n_bins, density=True, alpha=0.6, 
                             color='skyblue', label='Data', edgecolor='black')
                x_manual = np.linspace(data.min()*0.9, data.max()*1.1, 200)
                ax_manual.plot(x_manual, manual_dist.pdf(x_manual), 'red', 
                             linewidth=3, label='Manual Fit')
                ax_manual.legend()
                ax_manual.grid(True, alpha=0.3)
                st.pyplot(fig_manual)
                st.success("✅ Manual fit applied!")
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.markdown("---")
st.markdown("Beep Boop Histogram Fitter Boop Beep")

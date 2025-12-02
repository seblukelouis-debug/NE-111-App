import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

st.set_page_config(
    page_title="NE-111 Histogram Fitter", 
    page_icon="📊",
    layout="wide"
)

st.title("📊 Histogram Distribution Fitter")
st.markdown("**Fit SciPy distributions to your data** - Upload CSV or enter manually")

@st.cache_data
def parse_text_data(text: str) -> np.ndarray:
    text = text.replace("\n", ",").replace(";", ",").replace(" ", ",")
    arr = np.fromstring(text.strip(), sep=",")
    return arr[~np.isnan(arr)]

def load_csv_data(uploaded_file) -> tuple[np.ndarray, pd.DataFrame]:
    try:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found in CSV!")
            return np.array([]), df
        
        col_name = st.selectbox(
            "Select numeric column:", 
            numeric_cols, 
            key=f"col_{uploaded_file.name}"
        )
        data = df[col_name].dropna().values
        return data, df
    except Exception as e:
        st.error(f"CSV error: {str(e)}")
        return np.array([]), pd.DataFrame()

def compute_fit_errors(data: np.ndarray, dist_obj, bins: int = 30):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    pdf_vals = dist_obj.pdf(bin_centers)
    errors = np.abs(hist - pdf_vals)
    return np.mean(errors), np.max(errors), bin_centers, hist, pdf_vals

DIST_OPTIONS = {
    "Normal": stats.norm,
    "Gamma": stats.gamma, 
    "Exponential": stats.expon,
    "Lognormal": stats.lognorm,
    "Weibull": stats.weibull_min,
    "Beta": stats.beta,
    "Uniform": stats.uniform,
    "Chi-squared": stats.chi2,
    "Student-t": stats.t,
    "Cauchy": stats.cauchy,
    "Rayleigh": stats.rayleigh,
    "Logistic": stats.logistic
}

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📥 Data Input")
    
    input_type = st.radio("Data source:", ["Manual entry", "Upload CSV"], horizontal=True)
    
    data = np.array([])
    df = pd.DataFrame()
    
    if input_type == "Manual entry":
        example_data = "1.2, 2.1, 1.8, 3.0, 2.5, 1.9, 2.2"
        user_input = st.text_area(
            "Enter numbers (comma/space/newline separated):",
            value="",
            placeholder=example_data,
            height=100
        )
        if user_input.strip():
            data = parse_text_data(user_input)
            
    else:
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        if uploaded_file is not None:
            data, df = load_csv_data(uploaded_file)
    
    if data.size > 0:
        st.success(f"✅ Loaded **{data.size:,}** data points")
        st.metric("Mean", f"{np.mean(data):.3f}")
        st.metric("Std", f"{np.std(data):.3f}")
        st.metric("Min/Max", f"{np.min(data):.1f} / {np.max(data):.1f}")
    else:
        st.warning("👆 Enter data or upload CSV to start fitting")

    st.header("⚙️ Fit Options")
    selected_dists = st.multiselect(
        "Choose distributions to fit:",
        options=list(DIST_OPTIONS.keys()),
        default=["Normal", "Gamma", "Exponential"],
        max_selections=6
    )
    
    n_bins = st.slider("Histogram bins", 10, 100, 30, 5)

with col2:
    st.header("📈 Results")
    
    if data.size == 0 or len(selected_dists) == 0:
        st.info("⚠️ Select data and distributions above")
    else:
        fits = {}
        for dist_name in selected_dists:
            try:
                dist_class = DIST_OPTIONS[dist_name]
                params = dist_class.fit(data)
                dist_fitted = dist_class(*params)
                avg_err, max_err, _, _, _ = compute_fit_errors(data, dist_fitted, n_bins)
                
                fits[dist_name] = {
                    'params': params,
                    'dist': dist_fitted,
                    'avg_error': avg_err,
                    'max_error': max_err
                }
            except Exception as e:
                st.error(f"Failed to fit {dist_name}: {str(e)}")
        
        if fits:
            best_fit = min(fits.items(), key=lambda x: x[1]['avg_error'])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(data, bins=n_bins, density=True, alpha=0.5, 
                   color='lightblue', label='Data', edgecolor='black')
            
            x_range = np.linspace(data.min(), data.max(), 500)
            colors = plt.cm.tab10(np.linspace(0, 1, len(fits)))
            
            for i, (name, fit_info) in enumerate(fits.items()):
                pdf = fit_info['dist'].pdf(x_range)
                label = f"{name} (err={fit_info['avg_error']:.3f})"
                if name == best_fit[0]:
                    label += " ⭐ BEST"
                    lw = 3
                else:
                    lw = 2
                ax.plot(x_range, pdf, color=colors[i], linewidth=lw, label=label)
            
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_title("Distribution Fits vs Data Histogram")
            st.pyplot(fig)
            
            st.subheader(f"⭐ Best fit: {best_fit[0]}")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.hist(data, bins=n_bins, density=True, alpha=0.6, 
                    color='lightgreen', label='Data')
            best_dist = best_fit[1]['dist']
            pdf_best = best_dist.pdf(x_range)
            ax2.plot(x_range, pdf_best, 'red', linewidth=4, label='Best fit')
            ax2.set_xlabel("Value")
            ax2.set_ylabel("Density")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
            st.subheader("📋 Fit Quality Comparison")
            table_data = []
            for name, info in fits.items():
                table_data.append({
                    'Distribution': name,
                    'Avg Error': f"{info['avg_error']:.4f}",
                    'Max Error': f"{info['max_error']:.4f}",
                    'Parameters': ', '.join([f"{p:.3f}" for p in info['params']])
                })
            
            df_fits = pd.DataFrame(table_data)
            df_fits = df_fits.sort_values('Avg Error')
            st.dataframe(df_fits, use_container_width=True)

with st.expander("🔧 Manual Parameter Fitting", expanded=False):
    st.markdown("**Interactive sliders for any distribution**")
    manual_dist = st.selectbox("Choose distribution:", list(DIST_OPTIONS.keys()))
    
    if data.size > 0:
        dist_class = DIST_OPTIONS[manual_dist]
        try:
            auto_params = dist_class.fit(data)
            st.info(f"Auto-fit parameters: {tuple(np.round(auto_params, 3))}")
            
            st.subheader("Adjust parameters:")
            manual_params = []
            n_params = len(auto_params)
            
            col_params = st.columns(min(4, n_params))
            for i, param_val in enumerate(auto_params):
                with col_params[i % 4]:
                    min_p = float(param_val * 0.1) if param_val > 0 else float(param_val - 2)
                    max_p = float(param_val * 3) if param_val > 0 else float(param_val + 2)
                    p_slider = st.slider(
                        f"P{i+1}", 
                        min_p, max_p, 
                        float(param_val),
                        0.01,
                        key=f"manual_{manual_dist}_{i}"
                    )
                    manual_params.append(p_slider)
            
            if st.button("Apply Manual Fit", key="apply_manual"):
                manual_dist_obj = dist_class(*manual_params)
                avg_err, max_err, _, _, _ = compute_fit_errors(data, manual_dist_obj, n_bins)
                
                fig_manual, ax_manual = plt.subplots(figsize=(10, 6))
                ax_manual.hist(data, bins=n_bins, density=True, alpha=0.6, label='Data')
                x_manual = np.linspace(data.min(), data.max(), 500)
                ax_manual.plot(x_manual, manual_dist_obj.pdf(x_manual), 'purple', linewidth=4, label='Manual fit')
                ax_manual.legend()
                ax_manual.set_title(f"Manual Fit: {manual_dist}")
                st.pyplot(fig_manual)
                
                st.success(f"Manual fit errors: Avg={avg_err:.4f}, Max={max_err:.4f}")
                st.code(f"Parameters: {tuple(np.round(manual_params, 4))}")
                
        except Exception as e:
            st.error(f"Manual fitting error: {str(e)}")
    else:
        st.warning("Load data first!")

st.markdown("---")
st.markdown("*Built for NE-111 course project. Uses SciPy distributions & Streamlit.*")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

st.set_page_config(page_title="NE-111 Histogram Fitter", page_icon="📊", layout="wide")

st.title("Histogram Distribution Fitter")
st.markdown("Upload CSV or enter data manually to fit distributions")

def parse_text_data(text: str) -> np.ndarray:
    if not text or not text.strip():
        return np.array([])
    text = text.replace("\n", ",").replace(";", ",").replace(" ", ",")
    try:
        arr = np.fromstring(text.strip(), sep=",")
        return arr[~np.isnan(arr)]
    except:
        return np.array([])

def load_csv_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return np.array([]), df
        col_name = st.selectbox("Select column:", numeric_cols, key=f"col_{uploaded_file.name}")
        return df[col_name].dropna().values, df
    except:
        return np.array([]), pd.DataFrame()

def safe_fit(dist_class, data):
    try:
        if np.any(data <= 0):
            data_pos = data[data > 0]
            if len(data_pos) < 10:
                return None
            data = data_pos
        params = dist_class.fit(data)
        return dist_class(*params)
    except:
        return None

def compute_errors(data, dist_obj, bins=30):
    try:
        hist, edges = np.histogram(data, bins=bins, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        pdf_vals = dist_obj.pdf(centers)
        errors = np.abs(hist - pdf_vals)
        return np.mean(errors), np.max(errors)
    except:
        return np.inf, np.inf

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
}

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data Input")
    input_type = st.radio("Choose input:", ["Manual entry", "Upload CSV"], horizontal=True)
    
    data = np.array([])
    
    if input_type == "Manual entry":
        user_input = st.text_area("Enter numbers (comma/space/newline):", 
                                placeholder="1.2, 2.1, 1.8, 3.0, 2.5", height=100)
        data = parse_text_data(user_input)
    else:
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            data, _ = load_csv_data(uploaded)
    
    if len(data) > 0:
        st.success(f"Loaded {len(data)} points")
        col1m, col2m, col3m = st.columns(3)
        with col1m: st.metric("Mean", f"{np.mean(data):.3f}")
        with col2m: st.metric("Std", f"{np.std(data):.3f}")
        with col3m: st.metric("Range", f"{data.min():.1f}-{data.max():.1f}")
    else:
        st.warning("Enter data first")

    st.header("Options")
    selected_dists = st.multiselect("Distributions:", list(DIST_OPTIONS.keys()), 
                                  default=["Normal", "Gamma", "Exponential"])
    n_bins = st.slider("Bins", 10, 50, 25)

with col2:
    st.header("Histogram & Fits")
    
    if len(data) == 0:
        st.info("Enter data to see histogram")
    else:
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        ax_hist.hist(data, bins=n_bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title("Data Histogram")
        ax_hist.grid(True, alpha=0.3)
        st.pyplot(fig_hist)
        
        if len(selected_dists) > 0:
            fits = {}
            for name in selected_dists:
                dist = safe_fit(DIST_OPTIONS[name], data)
                if dist is not None:
                    avg_err, max_err = compute_errors(data, dist, n_bins)
                    fits[name] = {'dist': dist, 'avg_err': avg_err, 'max_err': max_err}
            
            if fits:
                fig_fit, ax_fit = plt.subplots(figsize=(10, 6))
                ax_fit.hist(data, bins=n_bins, density=True, alpha=0.5, color='lightblue', label='Data')
                
                x = np.linspace(data.min(), data.max(), 300)
                colors = plt.cm.Set1(np.linspace(0, 0.8, len(fits)))
                
                best_name, best_err = min(fits.items(), key=lambda x: x[1]['avg_err'])
                
                for i, (name, info) in enumerate(fits.items()):
                    pdf = info['dist'].pdf(x)
                    label = f"{name} (err={info['avg_err']:.3f})"
                    lw = 4 if name == best_name else 2
                    color = 'red' if name == best_name else colors[i]
                    ax_fit.plot(x, pdf, color=color, linewidth=lw, label=label)
                
                ax_fit.legend()
                ax_fit.set_title("Fitted Distributions")
                ax_fit.grid(True, alpha=0.3)
                st.pyplot(fig_fit)
                
                table_data = []
                for name, info in fits.items():
                    star = " (best)" if name == best_name else ""
                    table_data.append({
                        'Distribution': name + star,
                        'Avg Error': f"{info['avg_err']:.4f}",
                        'Max Error': f"{info['max_err']:.4f}"
                    })
                st.subheader("Fit Comparison")
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            else:
                st.warning("No distributions could be fitted to this data")

with st.expander("Manual Fitting"):
    if len(data) > 0:
        dist_name = st.selectbox("Distribution:", list(DIST_OPTIONS.keys()))
        dist_class = DIST_OPTIONS[dist_name]
        auto_fit = safe_fit(dist_class, data)
        
        if auto_fit:
            params = auto_fit.args
            st.info(f"Auto parameters: {tuple(np.round(params, 3))}")
            
            cols = st.columns(3)
            manual_params = []
            for i, p in enumerate(params[:3]):
                with cols[i]:
                    slider = st.slider(f"P{i+1}", 0.01, float(p*3), float(p), 0.01)
                    manual_params.append(slider)
            
            if st.button("Apply Manual Fit"):
                try:
                    manual_dist = dist_class(*manual_params)
                    fig_man, ax_man = plt.subplots(figsize=(10, 6))
                    ax_man.hist(data, bins=n_bins, density=True, alpha=0.6, label='Data')
                    x_man = np.linspace(data.min(), data.max(), 300)
                    ax_man.plot(x_man, manual_dist.pdf(x_man), 'purple', linewidth=4, label='Manual')
                    ax_man.legend()
                    st.pyplot(fig_man)
                except Exception as e:
                    st.error(f"Manual fit failed: {e}")

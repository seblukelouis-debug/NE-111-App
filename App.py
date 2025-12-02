import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

st.set_page_config(page_title="Histogram Fitter", layout="wide")

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

def load_csv_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return np.array([]), df
        col_name = st.selectbox("Select column:", list(numeric_cols))
        return df[col_name].dropna().values, df
    except:
        return np.array([]), pd.DataFrame()

DIST_OPTIONS = {
    "Normal": stats.norm,
    "Gamma": stats.gamma,
    "Exponential": stats.expon,
    "Lognormal": stats.lognorm,
    "Weibull": stats.weibull_min
}

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data Input")
    input_type = st.radio("Input method:", ["Manual", "CSV"])
    
    data = np.array([])
    
    if input_type == "Manual":
        user_input = st.text_area("Enter numbers:", placeholder="1.2, 2.1, 1.8, 3.0", height=80)
        data = parse_text_data(user_input)
    else:
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            data, _ = load_csv_data(uploaded)
    
    if len(data) > 0:
        st.success(f"Loaded {len(data)} points")
        st.metric("Mean", f"{np.mean(data):.2f}")
        st.metric("Std Dev", f"{np.std(data):.2f}")
    else:
        st.warning("Enter data to see plots")
    
    selected_dists = st.multiselect("Distributions", list(DIST_OPTIONS.keys()), 
                                   default=["Normal", "Gamma"])
    n_bins = st.slider("Number of bins", 10, 50, 20)

with col2:
    if len(data) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=n_bins, density=True, alpha=0.7, color='lightblue', edgecolor='black')
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title("Data Histogram")
        plt.grid(True, alpha=0.3)
        st.pyplot(plt.gcf())
        plt.close()
        
        if len(selected_dists) > 0:
            fits = {}
            for name in selected_dists:
                try:
                    dist_class = DIST_OPTIONS[name]
                    params = dist_class.fit(data)
                    dist_fitted = dist_class(*params)
                    fits[name] = dist_fitted
                except:
                    continue
            
            if fits:
                plt.figure(figsize=(10, 6))
                plt.hist(data, bins=n_bins, density=True, alpha=0.5, color='lightblue', label='Data')
                
                x = np.linspace(data.min(), data.max(), 200)
                colors = ['red', 'green', 'blue', 'orange', 'purple']
                
                best_err = float('inf')
                best_name = None
                for i, (name, dist) in enumerate(fits.items()):
                    pdf = dist.pdf(x)
                    plt.plot(x, pdf, colors[i], linewidth=2, label=name)
                    err = np.mean(np.abs(dist.pdf(data) - 0.1))  
                    if err < best_err:
                        best_err = err
                        best_name = name
                
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.title("Fitted Distributions")
                plt.legend()
                plt.grid(True, alpha=0.3)
                st.pyplot(plt.gcf())
                plt.close()
                
                st.subheader("Fit Results")
                table_data = []
                for name in fits.keys():
                    table_data.append({'Distribution': name, 'Status': 'Success'})
                st.table(pd.DataFrame(table_data))

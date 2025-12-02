import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="NE-111 Histogram Fitter", layout="wide")

st.title("Histogram Distribution Fitter")

@st.cache_data
def parse_text_data(text):
    text = text.replace("\n", ",").replace(";", ",").replace(" ", ",")
    arr = np.fromstring(text.strip(), sep=",")
    return arr[~np.isnan(arr)]

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
    "Cauchy": stats.cauchy
}

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data Input")
    input_type = st.radio("Data source:", ["Manual entry", "Upload CSV"])

    data = np.array([])
    if input_type == "Manual entry":
        user_input = st.text_area("Enter numbers:", placeholder="1.2, 2.1, 1.8")
        if user_input.strip():
            data = parse_text_data(user_input)
    else:
        uploaded_file = st.file_uploader("CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            data = df.iloc[:, 0].dropna().values

    if data.size > 0:
        st.success(f"Loaded {data.size} points")
        col1a, col2a, col3a = st.columns(3)
        col1a.metric("Mean", f"{np.mean(data):.3f}")
        col2a.metric("Std", f"{np.std(data):.3f}")
        col3a.metric("Range", f"{np.min(data):.1f}-{np.max(data):.1f}")

    selected_dists = st.multiselect(
        "Distributions:",
        list(DIST_OPTIONS.keys()),
        default=["Normal", "Gamma"]
    )

    n_bins = st.slider("Bins", 10, 50, 25)

with col2:
    if data.size > 0 and selected_dists:
        fits = {}

        for name in selected_dists:
            dist = DIST_OPTIONS[name]
            params = dist.fit(data)

            if name in ["Normal", "Exponential", "Uniform", "Cauchy"]:
                loc, scale = params[-2], params[-1]
                dist_fit = dist(loc=loc, scale=scale)

            elif name in ["Gamma", "Weibull", "Chi-squared"]:
                shape, loc, scale = params
                dist_fit = dist(shape, loc=loc, scale=scale)

            elif name == "Lognormal":
                shape, loc, scale = params
                dist_fit = stats.lognorm(s=shape, loc=loc, scale=scale)

            elif name == "Beta":
                a, b, loc, scale = params
                dist_fit = stats.beta(a, b, loc=loc, scale=scale)

            elif name == "Student-t":
                df_t, loc, scale = params
                dist_fit = stats.t(df_t, loc=loc, scale=scale)

            fits[name] = {"params": params, "dist": dist_fit}

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=n_bins,
                name="Data",
                histnorm="probability density"
            )
        )

        x = np.linspace(data.min(), data.max(), 400)
        colors = px.colors.qualitative.Set1

        for i, (name, fit) in enumerate(fits.items()):
            y = fit["dist"].pdf(x)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=name,
                    line=dict(color=colors[i % len(colors)])
                )
            )

        fig.update_layout(
            height=500,
            title="Distribution Fits",
            xaxis_title="Value",
            yaxis_title="Density"
        )
        st.plotly_chart(fig, use_container_width=True)

        table_data = []
        for name, fit in fits.items():
            table_data.append({
                "Distribution": name,
                "Parameters": ", ".join([f"{p:.3f}" for p in fit["params"]])
            })

        st.dataframe(pd.DataFrame(table_data))

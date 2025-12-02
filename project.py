import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------
# Import emojis
# ----------------------
bar_chart = '\n📊'
line_chart = '\n📈'
put_in = '\n📥'
warning_sign = '\n⚠'
hand = '\n🫳'
cele = '\n🥳'

# ----------------------
# Helper function
# ----------------------
def check_fits(hist_y, hist_x_centers, fitted_y, mode):
    res = np.sum((hist_y - fitted_y) ** 2)
    tot = np.sum((hist_y - np.mean(hist_y)) ** 2)
    r2 = 1 - res / tot if tot > 0 else float("N/A")

    if mode == 'display':
        if r2 != "N/A":
            if r2 > 0.9:
                st.success("Excellent fit!")
            elif r2 > 0.75:
                st.info("Pretty good fit!")
            elif r2 > 0.5:
                st.warning("The fit is okay, try another distribution?")
            else:
                st.error("This fit is not fitting.")
        else:
            st.info("R² not available for constant data.")
    else:
        return r2

# ----------------------
# Title
# ----------------------
st.title(bar_chart + " Distribution Fitting Lab")
st.caption("Explore probability distributions.")

# ----------------------
# DATA INPUT
# ----------------------
with st.expander(put_in + " Data Input", expanded=True):
    data_input_method = st.radio(
        "How would you like to provide your data?",
        ["Type/Paste Data", "Upload CSV"]
    )

    data = None
    numeric_df = None

    if data_input_method == "Type/Paste Data":
        raw = st.text_area(
            "Enter numbers separated by commas:",
            placeholder="12, 15, 19, 22, 22, 25, 30..."
        )
        if raw.strip():
            try:
                raw_data = raw.split(',')
                data = np.array([x for x in raw_data], dtype=float)
                st.success(f"Loaded {data.size} values from text input.")
            except:
                st.error(warning_sign + " Invalid numbers!")

    else:
        file = st.file_uploader(put_in + " Upload CSV:", type=["csv"])
        if file:
            try:
                df = pd.read_csv(file)
                numeric_df = df.select_dtypes(include=['number']).copy()

                if numeric_df.empty:
                    st.error(warning_sign + " No numeric columns found.")
                else:
                    data = numeric_df.to_numpy().flatten()
                    data = data[np.isfinite(data)]
                    st.success(f"Loaded {data.size} numeric values from CSV.")

            except Exception as e:
                st.error(warning_sign + f" Error reading CSV: {e}")

    if data is None:
        st.warning("Please enter or upload data to continue.")
        st.stop()

# ----------------------
# SIDEBAR — Show Numeric Table Only
# ----------------------
if data is not None:
    st.sidebar.subheader("Applicable Numeric Data")
    st.sidebar.dataframe(pd.DataFrame({"Values": data}), use_container_width=True)

# ----------------------
# Distribution List
# ----------------------
dist_options = {
    'Normal': stats.norm,
    'Exponential': stats.expon,
    'Gamma': stats.gamma,
    'Weibull': stats.weibull_min,
    'Lognormal': stats.lognorm,
    'Beta': stats.beta,
    'Chi-square': stats.chi2,
    'Rayleigh': stats.rayleigh,
    'Pareto': stats.pareto,
    'Cauchy': stats.cauchy,
}

# ----------------------
# Tabs
# ----------------------
tab1, tab2, tab3 = st.tabs(["Automatic Fitting", "Manual Fitting", "Fit Report"])

# ----------------------
# AUTOMATIC FITTING
# ----------------------
with tab1:
    st.header(bar_chart + " Automatic Fitting")

    selected = st.selectbox("Choose a distribution", list(dist_options.keys()))
    dist = dist_options[selected]

    params = dist.fit(data)

    # Histogram for auto (fixed bin count)
    BIN_COUNT_DEFAULT = 20
    hist_y_auto, hist_edges_auto = np.histogram(data, bins=BIN_COUNT_DEFAULT, density=True)
    hist_centers_auto = (hist_edges_auto[:-1] + hist_edges_auto[1:]) / 2
    auto_y_eval = dist.pdf(hist_centers_auto, *params)

    check_fits(hist_y_auto, hist_centers_auto, auto_y_eval, "display")

    x = np.linspace(np.min(data), np.max(data), 400)
    pdf = dist.pdf(x, *params)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=BIN_COUNT_DEFAULT, density=True,
            alpha=0.55, edgecolor="black", linewidth=0.6,
            color="#89CFF0")
    ax.plot(x, pdf, linewidth=2)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    st.pyplot(fig)

# ----------------------
# MANUAL FITTING
# ----------------------
with tab2:
    st.header(hand + " Manual Fitting")
    st.write("Tweak the distribution parameters & bin count to see changes.")

    # BIN COUNT SLIDER
    BIN_COUNT = st.slider(
        "BIN_COUNT",
        min_value=5,
        max_value=200,
        value=20,
        step=1
    )

    hist_y_manual, hist_edges_manual = np.histogram(data, bins=BIN_COUNT, density=True)
    hist_centers_manual = (hist_edges_manual[:-1] + hist_edges_manual[1:]) / 2

    param_list = ['loc', 'scale']
    manual_params = []

    data_min = float(np.min(data))
    data_max = float(np.max(data))
    span = data_max - data_min if data_max > data_min else 1

    slider_min = data_min - 0.5 * span
    slider_max = data_max + 0.5 * span

    for name in param_list:
        value = st.slider(
            f"{name}",
            min_value=float(slider_min),
            max_value=float(slider_max),
            value=float(slider_min)
        )
        manual_params.append(value)

    manual_y_eval = dist.pdf(hist_centers_manual, *manual_params)
    check_fits(hist_y_manual, hist_centers_manual, manual_y_eval, "display")

    manual_pdf = dist.pdf(np.linspace(np.min(data), np.max(data), 400), *manual_params)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(data, bins=BIN_COUNT, density=True,
             alpha=0.55, edgecolor="black", linewidth=0.6,
             color="#89CFF0")
    ax2.plot(np.linspace(np.min(data), np.max(data), 400), manual_pdf, linewidth=2)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    st.pyplot(fig2)

# ----------------------
# FIT REPORT
# ----------------------
with tab3:
    st.header(line_chart + " Fitting Report & Error Analysis")

    # --- Recompute histogram for auto and manual separately to avoid mismatch ---
    # Auto
    hist_y_auto, hist_edges_auto = np.histogram(data, bins=BIN_COUNT_DEFAULT, density=True)
    hist_centers_auto = (hist_edges_auto[:-1] + hist_edges_auto[1:]) / 2
    auto_y_eval = dist.pdf(hist_centers_auto, *params)
    auto_avg_error = float(np.mean(np.abs(hist_y_auto - auto_y_eval)))
    auto_max_error = float(np.max(np.abs(hist_y_auto - auto_y_eval)))
    auto_r2 = check_fits(hist_y_auto, hist_centers_auto, auto_y_eval, "do_not")

    # Manual
    hist_y_manual, hist_edges_manual = np.histogram(data, bins=BIN_COUNT, density=True)
    hist_centers_manual = (hist_edges_manual[:-1] + hist_edges_manual[1:]) / 2
    manual_y_eval = dist.pdf(hist_centers_manual, *manual_params)
    manual_avg_error = float(np.mean(np.abs(hist_y_manual - manual_y_eval)))
    manual_max_error = float(np.max(np.abs(hist_y_manual - manual_y_eval)))
    manual_r2 = check_fits(hist_y_manual, hist_centers_manual, manual_y_eval, "do_not")

    col1, col2 = st.columns(2)

    col1.subheader("Auto Fit Parameters")
    for i, p in enumerate(params):
        col1.write(f"param_{i+1}: {p:.6f}")

    col1.subheader("Auto Fit Quality")
    col1.write(f"Average Error: {auto_avg_error:.6f}")
    col1.write(f"Max Error: {auto_max_error:.6f}")
    col1.write(f"R²: {auto_r2}")

    col2.subheader("Manual Fit Parameters")
    for name, p in zip(param_list, manual_params):
        col2.write(f"{name}: {p:.6f}")

    col2.subheader("Manual Fit Quality")
    col2.write(f"Average Error: {manual_avg_error:.6f}")
    col2.write(f"Max Error: {manual_max_error:.6f}")
    col2.write(f"R²: {manual_r2}")

    st.success("Your report is ready! " + cele)

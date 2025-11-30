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
# Function
# ----------------------
def check_fits (hist_y, hist_x, fitted_y, mode):
    # compute R2
    res = np.sum((hist_y - fitted_y)**2)
    tot = np.sum((hist_y - np.mean(hist_y))**2)
    
    # protect 0 division
    r2 = 1 - res/tot if tot > 0 else float("N/A")
    
    # Info display
    if mode == 'display':
        if r2 > 0.9:
            st.success("Excellent fit!")
        elif r2 > 0.75:
            st.info("Pretty good fit!")
        elif r2 > 0.5:
            st.warning("The fit is okay, try another distribution?")
        else:
            st.error("This fit is not fitting.")
    elif mode == 'do_not':
        return r2

# ----------------------
# Title & Subtitle
# ----------------------
st.title(bar_chart + 'Distribution Fitting Lab')
st.caption('Explore probability distributions.')

# ----------------------
# Expander — DATA INPUT
# ----------------------
with st.expander(put_in + 'Data Input'):
    
    data_input_method = st.radio(
        'How would you like to provide your data?',
        ['Type/Paste Data', 'Upload CSV']
        )

    # Load data
    data = None

    # Input by hand
    if data_input_method == "Type/Paste Data":
        raw = st.text_area(
            'Enter numbers separated by commas: ',
            placeholder='12, 15, 19, 22, 22, 25, 30...'
            )
        if raw.strip():
            try:
                raw_data = raw.split(',')
                data = np.array([x for x in raw_data], dtype=float)
            except:
                st.error(warning_sign + ' Please enter valid numerical data!')
    # Input by uploading CSV file
    else:
        file = st.file_uploader(put_in + ' Upload your CSV:', type=["csv"])
            
        if file:
            df = pd.read_csv(file)
            numeric_df = df.select_dtypes(include=['number'])
            data = numeric_df.to_numpy().flatten()
            data = data[np.isfinite(data)]
            st.write('Applicable data in your file: ')
            
            container = st.container(height = 300)
            container.write(data)

    # Display warning sign if input invalid
    if data is None:
        st.warning(warning_sign + ' Please enter or upload your data to begin!')
        st.stop()

# ----------------------
# AVAILABLE DISTRIBUTIONS
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

# Tabs for layout
tab1, tab2, tab3 = st.tabs(['Automatic Fitting', 'Manual Fitting', 'Fit Report'])

# ----------------------
# AUTOMATIC FITTING
# ----------------------
with tab1:
    st.header(bar_chart + 'Automatic Fitting')
    box_name = list(dist_options.keys())

    selected = st.selectbox('Choose a distribution to fit', box_name)
    dist = dist_options[selected]

    # Perform fit using scipy
    params = dist.fit(data)

    # parameter names 
    param_list = ['loc', 'scale']

    # Compute auto probability density function
    x = np.linspace(min(data), max(data), 400)
    pdf = dist.pdf(x, *params)
    hist_y, hist_x = np.histogram(data, density=True)
    hist_centers = (hist_x[:-1] + hist_x[1:]) / 2
    auto_y = dist.pdf(hist_centers, *params)
    
    # check if the parameter fits the overall trend or not
    check_fits(hist_y, hist_x, auto_y, 'display')

    # Plot    
    fig, ax = plt.subplots()
    ax.hist(data, density=True, alpha=0.5)
    ax.plot(x, pdf, linewidth=2)
    ax.legend()
    st.pyplot(fig)
    
    st.success('Your Graph is Ready! ' + cele)

# ----------------------
# MANUAL FITTING
# ----------------------
with tab2:
    # Title brief summary
    st.header(hand + 'Manual Fitting')
    st.write('Tweak the parameters and see how the graph change!')

    # dynamic sliders
    manual_params = []
    slider_min = float(min(data)) - 5
    slider_max = float(max(data)) + 5
    for i, name in enumerate(param_list):
        slider = st.slider(
                f"{name}",
                min_value = slider_min,
                max_value = slider_max,
                value = slider_min
                )

        manual_params.append(slider)

    # Compute manual probability density function
    manual_pdf = dist.pdf(x, *manual_params)
    manual_params_values = manual_params 
    manual_y = dist.pdf(hist_centers, *manual_params_values)
    
    # Check if the parameter fits the overall trend
    check_fits(hist_y, hist_x, manual_y, 'display')

    # Plot
    fig2, ax2 = plt.subplots()
    ax2.hist(data, density=True, alpha=0.5)
    ax2.plot(x, manual_pdf, linewidth=2)    
    st.pyplot(fig2)
    
    st.success('Your Graph is Ready! ' + cele)

# ----------------------
# FIT REPORT
# ----------------------
with tab3:
    # Title
    st.header(line_chart + ' Fitting Report & Error Analysis')
    
    # Compute auto fit error
    auto_avg_error = np.mean(np.abs(hist_y - auto_y))
    auto_max_error = np.max(np.abs(hist_y - auto_y))
    auto_r2 = check_fits(hist_y, hist_x, auto_y, 'do_not')
    
    # Compute manual fit error
    manual_avg_error = np.mean(np.abs(hist_y - manual_y))
    manual_max_error = np.max(np.abs(hist_y - manual_y))
    manual_r2 = check_fits(hist_y, hist_x, manual_y, 'do_not')
    
    col1, col2 = st.columns(2,
                            vertical_alignment="top", 
                            border=True
                 )
    
    # Print auto fit parameter and fit error
    col1.subheader('Auto Fit Parameters')
    for name, p in zip(param_list, params):
        col1.write(f"**{name}:** {p:.4f}")

    col1.subheader('Automatic Fit Quality Metrics')
    col1.write(f'**Average Error:** {auto_avg_error:.6f}')
    col1.write(f'**Maximum Error:** {auto_max_error:.6f}')
    col1.write(f'**R^2 Value:** {auto_r2:.6f}')
    
    # Print manual fit parameter and fit error
    col2.subheader('Manual Fit Parameters')
    for name, p in zip(param_list, manual_params_values):
        col2.write(f"**{name}:** {p:.4f}")

    col2.subheader('Manual Fit Quality Metrics')
    col2.write(f"**Average Error:** {manual_avg_error:.6f}")
    col2.write(f"**Maximum Error:** {manual_max_error:.6f}")
    col2.write(f"**R^2:** {manual_r2:.6f}")
    
    # Display message if graph were generated successfully
    st.success('Your report is ready ' + cele)

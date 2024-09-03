import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt

# User input for the ticker symbol
ticker = st.text_input("Enter the ticker symbol", value="GGAL")

# User input for the SMA window
sma_window = st.number_input("Enter the SMA window (number of days)", min_value=1, value=21)

# User input for date range with default start date set to January 1, 2000
start_date = st.date_input(
    "Select the start date",
    value=pd.to_datetime('2000-01-01'),
    min_value=pd.to_datetime('1900-01-01'),
    max_value=pd.to_datetime('today')
)
end_date = st.date_input(
    "Select the end date",
    value=pd.to_datetime('today'),
    min_value=pd.to_datetime('1900-01-01'),
    max_value=pd.to_datetime('2024-08-31')
)

# User input for the close price type
close_price_type = st.selectbox("Select Close Price Type", ["Unadjusted", "Adjusted"])

# Option to use ratio adjustment
use_ratio = st.checkbox("Use YPFD.BA/YPF ratio adjustment")

# Fetch historical data for the specified ticker
data = yf.download(ticker, start=start_date, end=end_date)

if use_ratio:
    # Fetch data for YPFD.BA and YPF
    ypf_data = yf.download('YPF', start=start_date, end=end_date)
    ypfd_data = yf.download('YPFD.BA', start=start_date, end=end_date)
    
    # Reindex to match the dates
    all_dates = data.index.union(ypf_data.index).union(ypfd_data.index)
    data = data.reindex(all_dates)
    ypf_data = ypf_data.reindex(all_dates)
    ypfd_data = ypfd_data.reindex(all_dates)

    # Fill missing values with previous available data
    data = data.fillna(method='ffill')
    ypf_data = ypf_data.fillna(method='ffill')
    ypfd_data = ypfd_data.fillna(method='ffill')

    # Calculate the ratio
    ratio = ypfd_data['Close'] / ypf_data['Close']
    
    # Adjust the prices based on the ratio
    data[price_column] = data[price_column] / ratio

# Select close price based on user input
price_column = 'Adj Close' if close_price_type == "Adjusted" else 'Close'

# Calculate the user-defined SMA
sma_label = f'{sma_window}_SMA'
data[sma_label] = data[price_column].rolling(window=sma_window).mean()

# Calculate the dispersion (price - SMA)
data['Dispersion'] = data[price_column] - data[sma_label]

# Calculate the dispersion percentage
data['Dispersion_Percent'] = data['Dispersion'] / data[sma_label] * 100

# User input for the number of bins in the histogram
num_bins = st.slider("Select the number of bins for the histogram", min_value=10, max_value=100, value=50)

# User input for the color of the histogram
hist_color = st.color_picker("Pick a color for the histogram", value='#1f77b4')

# User input for colors of percentile lines
percentile_colors = {}
for percentile in percentiles:
    color = st.color_picker(f"Pick a color for the {percentile}th percentile line", value='red')
    percentile_colors[percentile] = color

# Plotly Histogram: Dispersion Percent with User Customization
fig_hist = go.Figure()

# Add the histogram trace
fig_hist.add_trace(
    go.Histogram(
        x=data['Dispersion_Percent'].dropna(),
        nbinsx=num_bins,
        marker_color=hist_color,
        opacity=0.75
    )
)

# Add percentile lines as vertical shapes
for percentile, value in zip(percentiles, percentile_values):
    fig_hist.add_vline(
        x=value,
        line=dict(color=percentile_colors.get(percentile, "red"), width=2, dash="dash"),
        annotation_text=f'{percentile}th percentile',
        annotation_position="top",
        annotation=dict(
            textangle=-90,  # Rotate text to vertical
            font=dict(color=percentile_colors.get(percentile, "red"))
        )
    )

# Update layout for interactivity and customization
fig_hist.update_layout(
    title=f"Customizable Histogram of Dispersion Percentage for {ticker} ({close_price_type})",
    xaxis_title="Dispersion (%)",
    yaxis_title="Frequency",
    bargap=0.1,  # Gap between bars
    template="plotly_dark",
    showlegend=False
)

# Display the interactive Plotly histogram
st.plotly_chart(fig_hist)

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# User input for the ticker symbol
ticker = st.text_input("Enter the ticker symbol", value="GGAL")

# User input for the SMA window
sma_window = st.number_input("Enter the SMA window (number of days)", min_value=1, value=21)

# User input for date range with default start date set to January 1, 2000
start_date = st.date_input("Select the start date", value=pd.to_datetime('2000-01-01'))
end_date = st.date_input("Select the end date", value=pd.to_datetime('today'))

# User input for close price type
close_price_type = st.selectbox("Select Close Price Type", ["Unadjusted", "Adjusted"])

# Checkbox for ratio adjustment
apply_ratio = st.checkbox("Divide by YPFD.BA/YPF ratio")

# Fetch historical data for the specified ticker
data = yf.download(ticker, start=start_date, end=end_date)

# Select close price based on user input
price_column = 'Adj Close' if close_price_type == "Adjusted" else 'Close'

if apply_ratio:
    # Fetch data for YPFD.BA and YPF
    ypfd_data = yf.download("YPFD.BA", start=start_date, end=end_date)[price_column]
    ypf_data = yf.download("YPF", start=start_date, end=end_date)[price_column]
    
    # Forward-fill missing data to handle unavailable dates
    ypfd_data.ffill(inplace=True)
    ypf_data.ffill(inplace=True)
    
    # Calculate the YPFD.BA/YPF ratio
    ratio = ypfd_data / ypf_data
    
    # Divide the original data by the ratio
    data[price_column] /= ratio

# Calculate the user-defined SMA
sma_label = f'{sma_window}_SMA'
data[sma_label] = data[price_column].rolling(window=sma_window).mean()

# Calculate the dispersion (price - SMA)
data['Dispersion'] = data[price_column] - data[sma_label]

# Calculate the dispersion percentage
data['Dispersion_Percent'] = data['Dispersion'] / data[sma_label] * 100

# Plotly Line Plot: Historical Price with SMA
fig = go.Figure()

# Plot the historical close price
fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name='Close Price'))

# Plot the SMA
fig.add_trace(go.Scatter(x=data.index, y=data[sma_label], mode='lines', name=f'{sma_window}-day SMA'))

# Update layout
fig.update_layout(
    title=f"Historical {close_price_type} Price of {ticker} with {sma_window}-day SMA",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    legend_title="Legend",
    template="plotly_dark"
)

# Show the Plotly chart
st.plotly_chart(fig)

# Plotly Line Plot: Historical Dispersion Percentage
fig_dispersion = go.Figure()

# Plot the dispersion percentage
fig_dispersion.add_trace(go.Scatter(x=data.index, y=data['Dispersion_Percent'], mode='lines', name='Dispersion %'))

# Add a red horizontal line at y=0
fig_dispersion.add_shape(
    go.layout.Shape(
        type="line",
        x0=data.index.min(),
        x1=data.index.max(),
        y0=0,
        y1=0,
        line=dict(color="red", width=2)
    )
)

# Update layout
fig_dispersion.update_layout(
    title=f"Historical Dispersion Percentage of {ticker} ({close_price_type})",
    xaxis_title="Date",
    yaxis_title="Dispersion (%)",
    legend_title="Legend",
    template="plotly_dark"
)

# Show the Plotly chart for dispersion percentage
st.plotly_chart(fig_dispersion)

# Seaborn/Matplotlib Histogram: Dispersion Percent with Percentiles
percentiles = [95, 75, 50, 25, 5]
percentile_values = np.percentile(data['Dispersion_Percent'].dropna(), percentiles)

plt.figure(figsize=(10, 6))
sns.histplot(data['Dispersion_Percent'].dropna(), kde=True, color='blue', bins=100)

# Add percentile lines
for percentile, value in zip(percentiles, percentile_values):
    plt.axvline(value, color='red', linestyle='--')
    plt.text(value, plt.ylim()[1]*0.9, f'{percentile}th', color='red')

plt.title(f'Dispersion Percentage of {ticker} ({close_price_type}) Close Price from {sma_window}-day SMA')
plt.xlabel('Dispersion (%)')
plt.ylabel('Frequency')
st.pyplot(plt)

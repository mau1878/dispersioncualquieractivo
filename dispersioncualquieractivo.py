# Import necessary modules
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
    max_value=pd.to_datetime('today')
)


# Ensure the start date is no later than today (Streamlit automatically restricts future dates)
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# User input for close price type
close_price_type = st.selectbox("Select Close Price Type", ["Unadjusted", "Adjusted"])


# Check if user wants to apply the ratio adjustment
apply_ratio = st.checkbox("Adjust price by YPFD.BA/YPF ratio")

# Fetch historical data for the specified ticker
data = yf.download(ticker, start=start_date, end=end_date)

if apply_ratio:
    # Fetch data for YPFD.BA and YPF
    ypfd_ba_data = yf.download("YPFD.BA", start=start_date, end=end_date)
    ypf_data = yf.download("YPF", start=start_date, end=end_date)

    # Forward fill to handle missing data (use the most recent previous value)
    ypfd_ba_data.ffill(inplace=True)
    ypf_data.ffill(inplace=True)

    # Calculate the ratio YPFD.BA/YPF
    ratio = ypfd_ba_data['Adj Close'] / ypf_data['Adj Close']

    # Adjust the original ticker's price by the ratio
    data['Adj Close'] = data['Adj Close'] / ratio
    data['Close'] = data['Close'] / ratio

# Select close price based on user input
price_column = 'Adj Close' if close_price_type == "Adjusted" else 'Close'

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
percentiles = [95, 85, 75, 50, 25, 15, 5]
percentile_values = np.percentile(data['Dispersion_Percent'].dropna(), percentiles)

plt.figure(figsize=(10, 6))
sns.histplot(data['Dispersion_Percent'].dropna(), kde=True, color='blue', bins=100)

# Add percentile lines
for percentile, value in zip(percentiles, percentile_values):
    plt.axvline(value, color='red', linestyle='--')
    plt.text(
        value, 
        plt.ylim()[1] * 0.9, 
        f'{percentile}th', 
        color='red',
        rotation='vertical',   # Rotate text vertically
        verticalalignment='center',  # Align vertically
        horizontalalignment='right'  # Align horizontally
    )


plt.title(f'Dispersion Percentage of {ticker} ({close_price_type}) Close Price from {sma_window}-day SMA')
plt.xlabel('Dispersion (%)')
plt.ylabel('Frequency')
st.pyplot(plt)

# User input for the number of bins in the histogram
num_bins = st.slider("Select the number of bins for the histogram", min_value=10, max_value=100, value=50)

# User input for the color of the histogram
hist_color = st.color_picker("Pick a color for the histogram", value='#1f77b4')

# Plotly Histogram: Dispersion Percent with User Customization
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

# Calculate mean and standard deviation for the bell curve
mean = np.mean(data['Dispersion_Percent'].dropna())
std_dev = np.std(data['Dispersion_Percent'].dropna())

# Generate x-values for the bell curve
x_values = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
# Calculate y-values for the bell curve
y_values = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / std_dev) ** 2)

# Add the bell curve to the plot
fig_hist.add_trace(
    go.Scatter(
        x=x_values,
        y=y_values * len(data['Dispersion_Percent'].dropna()) * (x_values[1] - x_values[0]),  # Scale y-values to match histogram
        mode='lines',
        line=dict(color='green', width=2),
        name='Bell Curve'
    )
)

# Add percentile lines as vertical shapes
for percentile, value in zip(percentiles, percentile_values):
    fig_hist.add_vline(
        x=value,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f'{percentile}th percentile',
        annotation_position="top",
        annotation=dict(
            textangle=-90,  # Rotate text to vertical
            font=dict(color="red")
        )
    )

# Update layout for interactivity and customization
fig_hist.update_layout(
    title=f"Customizable Histogram of Dispersion Percentage for {ticker} ({close_price_type})",
    xaxis_title="Dispersion (%)",
    yaxis_title="Frequency",
    bargap=0.1,  # Gap between bars
    template="plotly_dark",
    showlegend=True  # Ensure the legend is shown
)

# Display the interactive Plotly histogram
st.plotly_chart(fig_hist)

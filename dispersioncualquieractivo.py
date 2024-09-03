import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Entrada del usuario para el símbolo del ticker
ticker = st.text_input("Ingrese el símbolo del ticker", value="GGAL").upper()

# Entrada del usuario para la ventana de SMA
sma_window = st.number_input("Ingrese la ventana de SMA (número de días)", min_value=1, value=21)

# Entrada del usuario para el rango de fechas con fecha de inicio predeterminada al 1 de enero de 2000
start_date = st.date_input(
    "Seleccione la fecha de inicio",
    value=pd.to_datetime('2000-01-01'),
    min_value=pd.to_datetime('1900-01-01'),
    max_value=pd.to_datetime('today')
)
end_date = st.date_input(
    "Seleccione la fecha de fin",
    value=pd.to_datetime('today'),
    min_value=pd.to_datetime('1900-01-01'),
    max_value=pd.to_datetime('today')
)

# Asegúrese de que la fecha de inicio no sea posterior a hoy
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Entrada del usuario para el tipo de precio de cierre
close_price_type = st.selectbox("Seleccione el tipo de precio de cierre", ["No ajustado", "Ajustado"])

# Verificar si el usuario desea aplicar el ajuste por el ratio
apply_ratio = st.checkbox("Ajustar precio por el ratio YPFD.BA/YPF")

# Obtener datos históricos para el ticker especificado
data = yf.download(ticker, start=start_date, end=end_date)

if apply_ratio:
    # Obtener datos para YPFD.BA y YPF
    ypfd_ba_data = yf.download("YPFD.BA", start=start_date, end=end_date)
    ypf_data = yf.download("YPF", start=start_date, end=end_date)

    # Rellenar hacia adelante y hacia atrás para manejar datos faltantes
    ypfd_ba_data = ypfd_ba_data.ffill().bfill()
    ypf_data = ypf_data.ffill().bfill()

    # Calcular el ratio YPFD.BA/YPF
    ratio = ypfd_ba_data['Adj Close'] / ypf_data['Adj Close']

    # Sincronizar las fechas
    ratio = ratio.reindex(data.index).ffill()

    # Ajustar el precio del ticker original por el ratio
    data['Adj Close'] = data['Adj Close'] / ratio
    data['Close'] = data['Close'] / ratio

# Seleccionar el precio de cierre basado en la entrada del usuario
price_column = 'Adj Close' if close_price_type == "Ajustado" else 'Close'

# Calcular la SMA definida por el usuario
sma_label = f'SMA_{sma_window}'
data[sma_label] = data[price_column].rolling(window=sma_window).mean()

# Calcular la dispersión (precio - SMA)
data['Dispersión'] = data[price_column] - data[sma_label]

# Calcular el porcentaje de dispersión
data['Porcentaje_Dispersión'] = data['Dispersión'] / data[sma_label] * 100

# Gráfico de líneas con Plotly: Precio histórico con SMA
fig = go.Figure()

# Gráfico del precio de cierre histórico
fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name='Precio de Cierre'))

# Gráfico de la SMA
fig.add_trace(go.Scatter(x=data.index, y=data[sma_label], mode='lines', name=f'SMA de {sma_window} días'))

# Actualizar el diseño
fig.update_layout(
    title=f"Precio Histórico {close_price_type} de {ticker} con SMA de {sma_window} días",
    xaxis_title="Fecha",
    yaxis_title="Precio (USD)",
    legend_title="Leyenda",
    template="plotly_dark"
)

# Mostrar el gráfico de Plotly
st.plotly_chart(fig)

# Gráfico de líneas con Plotly: Porcentaje de dispersión histórico
fig_dispersion = go.Figure()

# Gráfico del porcentaje de dispersión
fig_dispersion.add_trace(go.Scatter(x=data.index, y=data['Porcentaje_Dispersión'], mode='lines', name='Porcentaje de Dispersión'))

# Añadir una línea horizontal roja en y=0
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

# Actualizar el diseño
fig_dispersion.update_layout(
    title=f"Porcentaje de Dispersión Histórico de {ticker} ({close_price_type})",
    xaxis_title="Fecha",
    yaxis_title="Dispersión (%)",
    legend_title="Leyenda",
    template="plotly_dark"
)

# Mostrar el gráfico de Plotly para el porcentaje de dispersión
st.plotly_chart(fig_dispersion)

# Histograma con Seaborn/Matplotlib: Porcentaje de dispersión con percentiles
percentiles = [95, 85, 75, 50, 25, 15, 5]
percentile_values = np.percentile(data['Porcentaje_Dispersión'].dropna(), percentiles)

plt.figure(figsize=(10, 6))
sns.histplot(data['Porcentaje_Dispersión'].dropna(), kde=True, color='blue', bins=100)

# Añadir líneas de percentiles
for percentile, value in zip(percentiles, percentile_values):
    plt.axvline(value, color='red', linestyle='--')
    plt.text(
        value, 
        plt.ylim()[1] * 0.9, 
        f'{percentile}th', 
        color='red',
        rotation='vertical',   # Rotar el texto verticalmente
        verticalalignment='center',  # Alinear verticalmente
        horizontalalignment='right'  # Alinear horizontalmente
    )

plt.title(f'Porcentaje de Dispersión de {ticker} ({close_price_type}) desde SMA de {sma_window} días')
plt.xlabel('Dispersión (%)')
plt.ylabel('Frecuencia')
st.pyplot(plt)

# Entrada del usuario para el número de bins en el histograma
num_bins = st.slider("Seleccione el número de bins para el histograma", min_value=10, max_value=100, value=50)

# Entrada del usuario para el color del histograma
hist_color = st.color_picker("Elija un color para el histograma", value='#1f77b4')

# Histograma con Plotly: Porcentaje de dispersión con personalización del usuario
fig_hist = go.Figure()

# Añadir la traza del histograma
fig_hist.add_trace(
    go.Histogram(
        x=data['Porcentaje_Dispersión'].dropna(),
        nbinsx=num_bins,
        marker_color=hist_color,
        opacity=0.75
    )
)

# Añadir líneas de percentiles como formas verticales
for percentile, value in zip(percentiles, percentile_values):
    fig_hist.add_vline(
        x=value,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f'{percentile}th percentile',
        annotation_position="top",
        annotation=dict(
            textangle=-90,  # Rotar el texto a vertical
            font=dict(color="red")
        )
    )

# Actualizar el diseño para interactividad y personalización
fig_hist.update_layout(
    title=f"Histograma Personalizable del Porcentaje de Dispersión para {ticker} ({close_price_type})",
    xaxis_title="Dispersión (%)",
    yaxis_title="Frecuencia",
    bargap=0.1,  # Espacio entre barras
    template="plotly_dark",
    showlegend=False
)

# Mostrar el histograma interactivo de Plotly
st.plotly_chart(fig_hist)

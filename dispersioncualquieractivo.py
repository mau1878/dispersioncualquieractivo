import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ===========================
# **Interfaz de Usuario (UI)**
# ===========================

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
  value=pd.to_datetime('today') + pd.DateOffset(days=1),  # Fecha de fin predeterminada: mañana
  min_value=pd.to_datetime('1900-01-01'),
  max_value=pd.to_datetime('today') + pd.DateOffset(days=1)  # Limitar a mañana
)

# Asegúrese de que la fecha de inicio no sea posterior a la fecha de fin
if start_date > end_date:
  st.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
  st.stop()

# Conversión a datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Entrada del usuario para el tipo de precio de cierre
close_price_type = st.selectbox("Seleccione el tipo de precio de cierre", ["No ajustado", "Ajustado"])

# Verificar si el usuario desea aplicar el ajuste por el ratio
apply_ratio = st.checkbox("Ajustar precio por el ratio YPFD.BA/YPF")

# ===========================
# **Obtención y Procesamiento de Datos**
# ===========================

@st.cache_data(show_spinner=False)
def get_data(ticker, start, end):
  data = yf.download(ticker, start=start, end=end)
  return data

# Obtener datos históricos para el ticker especificado
data = get_data(ticker, start_date, end_date)

# Verificar si se obtuvieron datos
if data.empty:
  st.error(f"No se encontraron datos para el ticker '{ticker}'. Por favor, verifique el símbolo e intente nuevamente.")
  st.stop()

if apply_ratio:
  # Obtener datos para YPFD.BA y YPF
  ypfd_ba_data = get_data("YPFD.BA", start_date, end_date)
  ypf_data = get_data("YPF", start_date, end_date)

  # Verificar si se obtuvieron datos para ambas acciones
  if ypfd_ba_data.empty or ypf_data.empty:
      st.error("No se pudieron obtener datos para 'YPFD.BA' o 'YPF'. Asegúrese de que los símbolos sean correctos.")
      st.stop()

  # Rellenar hacia adelante y hacia atrás para manejar datos faltantes
  ypfd_ba_data = ypfd_ba_data.ffill().bfill()
  ypf_data = ypf_data.ffill().bfill()

  # Calcular el ratio YPFD.BA/YPF
  ratio = ypfd_ba_data['Adj Close'] / ypf_data['Adj Close']

  # Sincronizar las fechas con el ticker principal
  ratio = ratio.reindex(data.index).ffill()

  # Ajustar el precio del ticker original por el ratio
  # Asegurarse de que 'Adj Close' y 'Close' son Series, no DataFrames
  if isinstance(data['Adj Close'], pd.DataFrame):
      data['Adj Close'] = data['Adj Close'].iloc[:, 0]
  if isinstance(data['Close'], pd.DataFrame):
      data['Close'] = data['Close'].iloc[:, 0]

  # Aplicar el ratio
  data['Adj Close'] = data['Adj Close'] / ratio
  data['Close'] = data['Close'] / ratio

# Seleccionar el precio de cierre basado en la entrada del usuario
price_column = 'Adj Close' if close_price_type == "Ajustado" else 'Close'

# Asegurarse de que el precio seleccionado es una Serie
if isinstance(data[price_column], pd.DataFrame):
  # Si es DataFrame, seleccionar la primera columna
  data[price_column] = data[price_column].iloc[:, 0]

# Calcular la SMA definida por el usuario
sma_label = f'SMA_{sma_window}'
data[sma_label] = data[price_column].rolling(window=sma_window).mean()

# Asegurarse de que 'sma_label' es una Serie y no un DataFrame
if isinstance(data[sma_label], pd.DataFrame):
  data[sma_label] = data[sma_label].iloc[:, 0]

# Calcular la dispersión (precio - SMA)
# Utilizar .squeeze() para asegurarse de que es una Serie
data['Dispersión'] = data[price_column].squeeze() - data[sma_label].squeeze()

# Calcular el porcentaje de dispersión
data['Porcentaje_Dispersión'] = (data['Dispersión'] / data[sma_label]) * 100

# Opcional: Depuración (descomentar si es necesario)
# st.write("Tipo de data[price_column]:", type(data[price_column]))
# st.write("Tipo de data[sma_label]:", type(data[sma_label]))
# st.write("Primeras filas de data[price_column]:", data[price_column].head())
# st.write("Primeras filas de data[sma_label]:", data[sma_label].head())

# ===========================
# **Visualizaciones**
# ===========================

# ---------------------------
# **Gráfico de Precio y SMA**
# ---------------------------
fig_price_sma = go.Figure()

# Precio de cierre histórico
fig_price_sma.add_trace(go.Scatter(
  x=data.index, 
  y=data[price_column], 
  mode='lines', 
  name='Precio de Cierre',
  line=dict(color='blue')
))

# SMA
fig_price_sma.add_trace(go.Scatter(
  x=data.index, 
  y=data[sma_label], 
  mode='lines', 
  name=f'SMA de {sma_window} días',
  line=dict(color='orange')
))

# Añadir watermark
fig_price_sma.add_annotation(
  text="MTaurus. X: mtaurus_ok",
  xref="paper", yref="paper",
  x=0.95, y=0.05,
  showarrow=False,
  font=dict(size=14, color="gray"),
  opacity=0.5
)

# Actualizar el diseño
fig_price_sma.update_layout(
  title=f"Precio Histórico {close_price_type} de {ticker} con SMA de {sma_window} días",
  xaxis_title="Fecha",
  yaxis_title="Precio (USD)",
  legend_title="Leyenda",
  template="plotly_dark",
  hovermode="x unified"
)

# Mostrar el gráfico
st.plotly_chart(fig_price_sma, use_container_width=True)

# ---------------------------
# **Gráfico de Porcentaje de Dispersión**
# ---------------------------
fig_dispersion = go.Figure()

# Porcentaje de dispersión
fig_dispersion.add_trace(go.Scatter(
  x=data.index, 
  y=data['Porcentaje_Dispersión'], 
  mode='lines', 
  name='Porcentaje de Dispersión',
  line=dict(color='green')
))

# Línea horizontal en y=0
fig_dispersion.add_shape(
  type="line",
  x0=data.index.min(),
  x1=data.index.max(),
  y0=0,
  y1=0,
  line=dict(color="red", width=2)
)

# Añadir watermark
fig_dispersion.add_annotation(
  text="MTaurus. X: mtaurus_ok",
  xref="paper", yref="paper",
  x=0.95, y=0.05,
  showarrow=False,
  font=dict(size=14, color="gray"),
  opacity=0.5
)

# Actualizar el diseño
fig_dispersion.update_layout(
  title=f"Porcentaje de Dispersión Histórico de {ticker} ({close_price_type})",
  xaxis_title="Fecha",
  yaxis_title="Dispersión (%)",
  legend_title="Leyenda",
  template="plotly_dark",
  hovermode="x unified"
)

# Mostrar el gráfico
st.plotly_chart(fig_dispersion, use_container_width=True)

# ---------------------------
# **Histograma con Seaborn/Matplotlib**
# ---------------------------

# Definir percentiles
percentiles = [95, 85, 75, 50, 25, 15, 5]
percentile_values = np.percentile(data['Porcentaje_Dispersión'].dropna(), percentiles)

# Crear figura
plt.figure(figsize=(10, 6))
sns.histplot(
  data['Porcentaje_Dispersión'].dropna(), 
  kde=True, 
  color='blue', 
  bins=100
)

# Añadir líneas de percentiles
for percentile, value in zip(percentiles, percentile_values):
  plt.axvline(value, color='red', linestyle='--')
  plt.text(
      value, 
      plt.ylim()[1] * 0.9, 
      f'{percentile}th', 
      color='red',
      rotation=90,   # Rotar el texto verticalmente
      verticalalignment='center',  # Alinear verticalmente
      horizontalalignment='right'  # Alinear horizontalmente
  )

# Añadir watermark
plt.text(
  0.95, 0.05, "MTaurus. X: mtaurus_ok", 
  fontsize=14, color='gray', ha='right', va='center', alpha=0.5, transform=plt.gcf().transFigure
)

# Añadir títulos y etiquetas
plt.title(f'Porcentaje de Dispersión de {ticker} ({close_price_type}) desde SMA de {sma_window} días')
plt.xlabel('Dispersión (%)')
plt.ylabel('Frecuencia')

# Mostrar el histograma
st.pyplot(plt)
plt.close()  # Cerrar la figura para liberar memoria

# ---------------------------
# **Histograma con Plotly (Personalizado por el Usuario)**
# ---------------------------

# Entrada del usuario para el número de bins en el histograma
num_bins = st.slider(
  "Seleccione el número de bins para el histograma", 
  min_value=10, 
  max_value=100, 
  value=50
)

# Entrada del usuario para el color del histograma
hist_color = st.color_picker(
  "Elija un color para el histograma", 
  value='#1f77b4'
)

# Crear figura de histograma
fig_hist = go.Figure()

# Añadir traza del histograma
fig_hist.add_trace(
  go.Histogram(
      x=data['Porcentaje_Dispersión'].dropna(),
      nbinsx=num_bins,
      marker_color=hist_color,
      opacity=0.75,
      name='Distribución'
  )
)

# Añadir líneas de percentiles
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

# Añadir watermark
fig_hist.add_annotation(
  text="MTaurus. X: mtaurus_ok",
  xref="paper", yref="paper",
  x=0.95, y=0.05,
  showarrow=False,
  font=dict(size=14, color="gray"),
  opacity=0.5
)

# Actualizar el diseño para interactividad y personalización
fig_hist.update_layout(
  title=f'Histograma del Porcentaje de Dispersión de {ticker} ({close_price_type})',
  xaxis_title='Dispersión (%)',
  yaxis_title='Frecuencia',
  bargap=0.1,
  template="plotly_dark",
  hovermode="x unified",
  legend_title="Leyenda"
)

# Mostrar el histograma
st.plotly_chart(fig_hist, use_container_width=True)

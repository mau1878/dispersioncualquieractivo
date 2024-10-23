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

st.set_page_config(page_title="Análisis de Dispersión de Activos", layout="wide")

# Título de la aplicación
st.title("Análisis de Dispersión de Activos Financieros")

# Entrada del usuario para el símbolo del ticker
ticker = st.text_input("Ingrese el símbolo del ticker", value="GGAL").upper()

# Entrada del usuario para la ventana de SMA
sma_window = st.number_input("Ingrese la ventana de SMA (número de días)", min_value=1, value=21, step=1)

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
def get_data(ticker_symbol, start, end):
  try:
      data = yf.download(ticker_symbol, start=start, end=end)
      return data
  except Exception as e:
      st.error(f"Error al descargar datos para {ticker_symbol}: {e}")
      return pd.DataFrame()  # Retorna un DataFrame vacío en caso de error

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

  # Verificar que 'Adj Close' esté presente
  if 'Adj Close' not in ypfd_ba_data.columns or 'Adj Close' not in ypf_data.columns:
      st.error("Los datos descargados no contienen la columna 'Adj Close'.")
      st.stop()

  # Calcular el ratio YPFD.BA/YPF
  ratio = ypfd_ba_data['Adj Close'] / ypf_data['Adj Close']

  # Alinear el ratio con el índice de 'data'
  ratio_aligned, data_aligned = ratio.align(data.index.to_series(), join='right')

  # Verificar si el ratio alineado tiene la misma longitud que 'data'
  if len(ratio_aligned) != len(data):
      st.error("El ratio YPFD.BA/YPF no pudo alinearse correctamente con los datos del ticker principal.")
      st.stop()

  # Manejar posibles NaN en el ratio
  if ratio_aligned.isnull().any():
      st.warning("Se encontraron valores NaN en el ratio YPFD.BA/YPF. Estos se rellenarán hacia adelante y hacia atrás.")
      ratio_aligned = ratio_aligned.ffill().bfill()

  # Verificar nuevamente que no haya NaN
  if ratio_aligned.isnull().any():
      st.error("El ratio YPFD.BA/YPF contiene valores NaN incluso después de rellenar.")
      st.stop()

  # Aplicar el ratio
  try:
      data['Adj Close'] = data['Adj Close'] / ratio_aligned
      data['Close'] = data['Close'] / ratio_aligned
  except Exception as e:
      st.error(f"Error al aplicar el ratio YPFD.BA/YPF: {e}")
      st.stop()

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

# Limpiar datos: eliminar filas con NaN en las columnas cruciales
data_clean = data.dropna(subset=[price_column, sma_label, 'Dispersión', 'Porcentaje_Dispersión'])

# Verificar si hay datos después de limpiar
if data_clean.empty:
  st.error("Después de aplicar filtros y cálculos, no hay datos disponibles para visualizar.")
  st.stop()

# ===========================
# **Visualizaciones**
# ===========================

# ---------------------------
# **Gráfico de Precio y SMA**
# ---------------------------
fig_price_sma = go.Figure()

# Precio de cierre histórico
fig_price_sma.add_trace(go.Scatter(
  x=data_clean.index, 
  y=data_clean[price_column], 
  mode='lines', 
  name='Precio de Cierre',
  line=dict(color='blue')
))

# SMA
fig_price_sma.add_trace(go.Scatter(
  x=data_clean.index, 
  y=data_clean[sma_label], 
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
  hovermode="x unified",
  height=600
)

# Verificar que hay datos para graficar
if data_clean[price_column].empty:
  st.warning("No hay datos disponibles para graficar el Precio de Cierre.")
else:
  # Mostrar el gráfico
  st.plotly_chart(fig_price_sma, use_container_width=True)

# ---------------------------
# **Gráfico de Porcentaje de Dispersión**
# ---------------------------
fig_dispersion = go.Figure()

# Porcentaje de dispersión
fig_dispersion.add_trace(go.Scatter(
  x=data_clean.index, 
  y=data_clean['Porcentaje_Dispersión'], 
  mode='lines', 
  name='Porcentaje de Dispersión',
  line=dict(color='green')
))

# Línea horizontal en y=0
fig_dispersion.add_shape(
  type="line",
  x0=data_clean.index.min(),
  x1=data_clean.index.max(),
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
  hovermode="x unified",
  height=600
)

# Verificar que hay datos para graficar
if data_clean['Porcentaje_Dispersión'].empty:
  st.warning("No hay datos disponibles para graficar el Porcentaje de Dispersión.")
else:
  # Mostrar el gráfico
  st.plotly_chart(fig_dispersion, use_container_width=True)

# ---------------------------
# **Histograma con Seaborn/Matplotlib**
# ---------------------------

# Definir percentiles
percentiles = [95, 85, 75, 50, 25, 15, 5]
percentile_values = np.percentile(data_clean['Porcentaje_Dispersión'], percentiles)

# Crear figura
plt.figure(figsize=(10, 6))
sns.histplot(
  data_clean['Porcentaje_Dispersión'], 
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
      x=data_clean['Porcentaje_Dispersión'],  # Usar data_clean
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
  legend_title="Leyenda",
  height=600
)

# Verificar que hay datos para graficar
if data_clean['Porcentaje_Dispersión'].empty:
  st.warning("No hay datos disponibles para graficar el histograma de dispersión.")
else:
  # Mostrar el histograma
  st.plotly_chart(fig_hist, use_container_width=True)

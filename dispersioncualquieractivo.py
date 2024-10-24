import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configuración de la página de Streamlit
st.set_page_config(
  page_title="Análisis de SMA y Dispersión de Precios",
  layout="wide",
  initial_sidebar_state="expanded",
)

# Definir una función para aplanar las columnas MultiIndex
def flatten_columns(df):
  if isinstance(df.columns, pd.MultiIndex):
      df.columns = [' '.join(col).strip() for col in df.columns.values]
  return df

# Función para descargar datos y manejar MultiIndex
def download_data(ticker, start, end):
  try:
      df = yf.download(ticker, start=start, end=end)
      if df.empty:
          st.warning(f"No hay datos disponibles para el ticker **{ticker}** en el rango de fechas seleccionado.")
          return None
      df = flatten_columns(df)
      st.write(f"**Columnas para {ticker}:** {df.columns.tolist()}")
      return df
  except Exception as e:
      st.error(f"Error al descargar datos para el ticker **{ticker}**: {e}")
      return None

# Título de la aplicación
st.title("📈 Análisis de SMA y Dispersión de Precios - MTaurus")
st.markdown("### 🚀 Sigue nuestro trabajo en [Twitter](https://twitter.com/MTaurus_ok)")

# Entradas del usuario
ticker = st.text_input("🖊️ Ingrese el símbolo del ticker", value="GGAL").upper()

# Verificar si el ticker principal ha sido ingresado
if ticker:
  # Entrada del usuario para la ventana de SMA
  sma_window = st.number_input("📊 Ingrese la ventana de SMA (número de días)", min_value=1, value=21)

  # Entrada del usuario para el rango de fechas
  start_date = st.date_input(
      "📅 Seleccione la fecha de inicio",
      value=pd.to_datetime('2000-01-01'),
      min_value=pd.to_datetime('1900-01-01'),
      max_value=pd.to_datetime('today')
  )
  end_date = st.date_input(
      "📅 Seleccione la fecha de fin",
      value=pd.to_datetime('today') + pd.DateOffset(days=1),  # Establecer la fecha de fin predeterminada como el día siguiente a hoy
      min_value=pd.to_datetime('1900-01-01'),
      max_value=pd.to_datetime('today') + pd.DateOffset(days=1)  # Limitar la fecha de fin a "el día siguiente a hoy"
  )

  # Convertir las fechas a datetime
  start_date = pd.to_datetime(start_date)
  end_date = pd.to_datetime(end_date)

  # Validar que la fecha de inicio no sea posterior a la fecha de fin
  if start_date > end_date:
      st.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
  else:
      # Entrada del usuario para el tipo de precio de cierre
      close_price_type = st.selectbox("📈 Seleccione el tipo de precio de cierre", ["No ajustado", "Ajustado"])

      # Verificar si el usuario desea aplicar el ajuste por el ratio
      apply_ratio = st.checkbox("🔄 Ajustar precio por el ratio YPFD.BA/YPF")

      # Descargar datos para el ticker principal
      data = download_data(ticker, start_date, end_date)

      if data is not None:
          # Si se desea aplicar el ratio, descargar datos adicionales
          if apply_ratio:
              st.subheader("🔄 Aplicando ajuste por ratio YPFD.BA/YPF")
              ypfd_ba_ticker = "YPFD.BA"
              ypf_ticker = "YPF"

              ypfd_ba_data = download_data(ypfd_ba_ticker, start_date, end_date)
              ypf_data = download_data(ypf_ticker, start_date, end_date)

              if ypfd_ba_data is not None and ypf_data is not None:
                  # Asegurarse de que 'Adj Close' existe en ambos DataFrames
                  if 'Adj Close' in ypfd_ba_data.columns and 'Adj Close' in ypf_data.columns:
                      # Rellenar valores faltantes
                      ypfd_ba_data = ypfd_ba_data.fillna(method='ffill').fillna(method='bfill')
                      ypf_data = ypf_data.fillna(method='ffill').fillna(method='bfill')

                      # Calcular el ratio YPFD.BA/YPF
                      ratio = ypfd_ba_data['Adj Close'] / ypf_data['Adj Close']

                      # Sincronizar las fechas con el ticker principal
                      ratio = ratio.reindex(data.index).fillna(method='ffill').fillna(method='bfill')

                      # Ajustar el precio del ticker original por el ratio
                      if 'Adj Close' in data.columns:
                          data['Adj Close Ajustado'] = data['Adj Close'] / ratio
                      else:
                          st.warning(f"No se encontró la columna 'Adj Close' en el ticker **{ticker}**.")

                      if 'Close' in data.columns:
                          data['Close Ajustado'] = data['Close'] / ratio
                      else:
                          st.warning(f"No se encontró la columna 'Close' en el ticker **{ticker}**.")
                  else:
                      st.error(f"No se encontraron las columnas 'Adj Close' en **{ypfd_ba_ticker}** o **{ypf_ticker}**.")
              else:
                  st.error("No se pudieron descargar los datos necesarios para aplicar el ratio.")
          else:
              # Si no se aplica el ratio, asegurar que las columnas están correctamente nombradas
              if 'Adj Close' in data.columns:
                  data['Adj Close Original'] = data['Adj Close']
              if 'Close' in data.columns:
                  data['Close Original'] = data['Close']

          # Seleccionar el precio de cierre basado en la entrada del usuario
          if apply_ratio:
              price_column = 'Adj Close Ajustado' if close_price_type == "Ajustado" and 'Adj Close Ajustado' in data.columns else 'Adj Close'
              if apply_ratio and close_price_type == "No ajustado" and 'Adj Close Original' in data.columns:
                  price_column = 'Close Ajustado' if 'Close Ajustado' in data.columns else 'Close'
          else:
              price_column = 'Adj Close' if close_price_type == "Ajustado" else 'Close'

          # Verificar que la columna seleccionada existe
          if price_column not in data.columns:
              st.error(f"La columna seleccionada **{price_column}** no existe en los datos.")
          else:
              # Calcular la SMA definida por el usuario
              sma_label = f'SMA_{sma_window}'
              data[sma_label] = data[price_column].rolling(window=sma_window).mean()

              # Calcular la dispersión (precio - SMA)
              data['Dispersión'] = data[price_column] - data[sma_label]

              # Calcular el porcentaje de dispersión
              data['Porcentaje_Dispersión'] = (data['Dispersión'] / data[sma_label]) * 100

              # -------------------------------
              # Visualizaciones y Análisis
              # -------------------------------

              # 1. Gráfico de líneas con Plotly: Precio histórico con SMA
              st.write("### 📈 Precio Histórico con SMA")
              fig = go.Figure()

              # Gráfico del precio de cierre histórico
              fig.add_trace(go.Scatter(
                  x=data.index,
                  y=data[price_column],
                  mode='lines',
                  name='Precio de Cierre'
              ))

              # Gráfico de la SMA
              fig.add_trace(go.Scatter(
                  x=data.index,
                  y=data[sma_label],
                  mode='lines',
                  name=f'SMA de {sma_window} días'
              ))

              # Añadir watermark
              fig.add_annotation(
                  text="MTaurus. X: mtaurus_ok",
                  xref="paper", yref="paper",
                  x=0.95, y=0.05,
                  showarrow=False,
                  font=dict(size=14, color="gray"),
                  opacity=0.5
              )

              # Actualizar el diseño
              fig.update_layout(
                  title=f"Precio Histórico {'Ajustado' if close_price_type == 'Ajustado' else 'No Ajustado'} de {ticker} con SMA de {sma_window} días",
                  xaxis_title="Fecha",
                  yaxis_title="Precio (USD)",
                  legend_title="Leyenda",
                  template="plotly_dark",
                  hovermode="x unified"
              )

              # Mostrar el gráfico de Plotly
              st.plotly_chart(fig, use_container_width=True)

              # 2. Gráfico de líneas con Plotly: Porcentaje de dispersión histórico
              st.write("### 📉 Porcentaje de Dispersión Histórico")
              fig_dispersion = go.Figure()

              # Gráfico del porcentaje de dispersión
              fig_dispersion.add_trace(go.Scatter(
                  x=data.index,
                  y=data['Porcentaje_Dispersión'],
                  mode='lines',
                  name='Porcentaje de Dispersión'
              ))

              # Añadir una línea horizontal roja en y=0
              fig_dispersion.add_shape(
                  type="line",
                  x0=data.index.min(),
                  x1=data.index.max(),
                  y0=0,
                  y1=0,
                  line=dict(color="red", width=2),
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

              # Mostrar el gráfico de Plotly para el porcentaje de dispersión
              st.plotly_chart(fig_dispersion, use_container_width=True)

              # 3. Histograma con Seaborn/Matplotlib: Porcentaje de dispersión con percentiles
              st.write("### 📊 Histograma de Porcentaje de Dispersión con Percentiles")
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
                      f'{percentile}º Percentil', 
                      color='red',
                      rotation='vertical',   # Rotar el texto verticalmente
                      verticalalignment='center',  # Alinear verticalmente
                      horizontalalignment='right'  # Alinear horizontalmente
                  )

              # Añadir watermark
              plt.text(
                  0.95, 0.05, "MTaurus. X: mtaurus_ok", 
                  fontsize=14, color='gray', ha='right', va='center', alpha=0.5, transform=plt.gcf().transFigure
              )

              plt.title(f'Porcentaje de Dispersión de {ticker} ({close_price_type}) desde SMA de {sma_window} días')
              plt.xlabel('Dispersión (%)')
              plt.ylabel('Frecuencia')
              plt.tight_layout()
              st.pyplot(plt)
              plt.clf()  # Limpiar la figura para evitar superposiciones

              # 4. Entrada del usuario para el número de bins en el histograma
              st.write("### 🎨 Personalización del Histograma")
              num_bins = st.slider("Seleccione el número de bins para el histograma", min_value=10, max_value=100, value=50, key="bins_slider")
              hist_color = st.color_picker("Elija un color para el histograma", value='#1f77b4', key="color_picker")

              # Histograma con Plotly: Porcentaje de dispersión con personalización del usuario
              st.write("### 📊 Histograma de Porcentaje de Dispersión")
              fig_hist = go.Figure()

              # Añadir la traza del histograma
              fig_hist.add_trace(
                  go.Histogram(
                      x=data['Porcentaje_Dispersión'].dropna(),
                      nbinsx=num_bins,
                      marker_color=hist_color,
                      opacity=0.75,
                      name="Histograma"
                  )
              )

              # Añadir líneas de percentiles como formas verticales
              for percentile, value in zip(percentiles, percentile_values):
                  fig_hist.add_vline(
                      x=value,
                      line=dict(color="red", width=2, dash="dash"),
                      annotation_text=f'{percentile}º Percentil',
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
                  hovermode="x unified"
              )

              # Mostrar el gráfico de Plotly para el histograma
              st.plotly_chart(fig_hist, use_container_width=True)

else:
  st.warning("Por favor, ingrese un símbolo de ticker válido para comenzar el análisis.")

# Información adicional o footer (opcional)
st.markdown("---")
st.markdown("© 2024 MTaurus. Todos los derechos reservados.")

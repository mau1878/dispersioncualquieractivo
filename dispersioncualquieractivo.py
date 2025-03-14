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

# Función para aplanar las columnas MultiIndex
def flatten_columns(df, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        # Aplanar MultiIndex (Price, Ticker) en "Price Ticker"
        df.columns = [f"{col[0]} {col[1]}" for col in df.columns]
    else:
        # Si no es MultiIndex, añadir sufijo del ticker
        df.columns = [f"{col} {ticker}" for col in df.columns]
    return df

# Función para descargar datos y manejar MultiIndex
def download_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            st.warning(f"No hay datos disponibles para el ticker **{ticker}** en el rango de fechas seleccionado.")
            return None
        df = flatten_columns(df, ticker)
        st.write(f"**Columnas para {ticker}:** {df.columns.tolist()}")
        # st.write(df.head())  # Uncomment for debugging
        return df
    except Exception as e:
        st.error(f"Error al descargar datos para el ticker **{ticker}**: {e}")
        return None

# Título de la aplicación
st.title("📈 Análisis de SMA y Dispersión de Precios - MTaurus")
st.markdown("### 🚀 Sigue nuestro trabajo en [Twitter](https://twitter.com/MTaurus_ok)")

# Entradas del usuario
ticker = st.text_input("🖊️ Ingrese el símbolo del ticker", value="GGAL").upper()

if ticker:
    sma_window = st.number_input("📊 Ingrese la ventana de SMA (número de días)", min_value=1, value=21)
    start_date = st.date_input(
        "📅 Seleccione la fecha de inicio",
        value=pd.to_datetime('2000-01-01'),
        min_value=pd.to_datetime('1900-01-01'),
        max_value=pd.to_datetime('today')
    )
    end_date = st.date_input(
        "📅 Seleccione la fecha de fin",
        value=pd.to_datetime('today') + pd.DateOffset(days=1),
        min_value=pd.to_datetime('1900-01-01'),
        max_value=pd.to_datetime('today') + pd.DateOffset(days=1)
    )

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if start_date > end_date:
        st.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
    else:
        close_price_type = st.selectbox("📈 Seleccione el tipo de precio de cierre", ["No ajustado", "Ajustado"])
        apply_ratio = st.checkbox("🔄 Ajustar precio por el ratio YPFD.BA/YPF")

        # Descargar datos para el ticker principal
        data = download_data(ticker, start_date, end_date)

        if data is not None:
            # Definir columnas esperadas para el ticker principal
            adj_close_col_main = f"Adj Close {ticker}"
            close_col_main = f"Close {ticker}"

            if apply_ratio:
                st.subheader("🔄 Aplicando ajuste por ratio YPFD.BA/YPF")
                ypfd_ba_ticker = "YPFD.BA"
                ypf_ticker = "YPF"

                ypfd_ba_data = download_data(ypfd_ba_ticker, start_date, end_date)
                ypf_data = download_data(ypf_ticker, start_date, end_date)

                if ypfd_ba_data is not None and ypf_data is not None:
                    # Definir columnas para el ratio
                    adj_close_col_ypfd = f"Adj Close {ypfd_ba_ticker}"
                    adj_close_col_ypf = f"Adj Close {ypf_ticker}"
                    close_col_ypfd = f"Close {ypfd_ba_ticker}"
                    close_col_ypf = f"Close {ypf_ticker}"

                    # Elegir columnas para el ratio (preferir Adj Close, fallback a Close)
                    ypfd_price_col = adj_close_col_ypfd if adj_close_col_ypfd in ypfd_ba_data.columns else close_col_ypfd
                    ypf_price_col = adj_close_col_ypf if adj_close_col_ypf in ypf_data.columns else close_col_ypf

                    if ypfd_price_col in ypfd_ba_data.columns and ypf_price_col in ypf_data.columns:
                        ypfd_ba_data = ypfd_ba_data.fillna(method='ffill').fillna(method='bfill')
                        ypf_data = ypf_data.fillna(method='ffill').fillna(method='bfill')
                        ratio = ypfd_ba_data[ypfd_price_col] / ypf_data[ypf_price_col]
                        ratio = ratio.reindex(data.index).fillna(method='ffill').fillna(method='bfill')

                        # Ajustar el precio del ticker principal
                        if adj_close_col_main in data.columns:
                            data['Adj Close Ajustado'] = data[adj_close_col_main] / ratio
                        else:
                            st.warning(f"No se encontró 'Adj Close' para {ticker}. Usando 'Close' para ajuste.")
                            data['Adj Close Ajustado'] = data[close_col_main] / ratio
                        data['Close Ajustado'] = data[close_col_main] / ratio
                    else:
                        st.error(f"No se encontraron columnas de precio válidas para {ypfd_ba_ticker} o {ypf_ticker}.")
                else:
                    st.error("No se pudieron descargar los datos necesarios para aplicar el ratio.")
            else:
                # Sin ajuste por ratio
                if adj_close_col_main in data.columns:
                    data['Adj Close Original'] = data[adj_close_col_main]
                else:
                    st.warning(f"No se encontró 'Adj Close' para {ticker}. Usando 'Close'.")
                    data['Adj Close Original'] = data[close_col_main]
                data['Close Original'] = data[close_col_main]

            # Seleccionar el precio de cierre basado en la entrada del usuario
            if apply_ratio:
                price_column = (
                    'Adj Close Ajustado' if (close_price_type == "Ajustado" and 'Adj Close Ajustado' in data.columns)
                    else 'Close Ajustado' if 'Close Ajustado' in data.columns
                    else close_col_main
                )
            else:
                price_column = (
                    'Adj Close Original' if (close_price_type == "Ajustado" and 'Adj Close Original' in data.columns)
                    else 'Close Original' if 'Close Original' in data.columns
                    else close_col_main
                )

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

                # [Insert your visualization code here]
                # Example:
                st.write("### 📈 Precio Histórico con SMA")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name='Precio de Cierre'))
                fig.add_trace(go.Scatter(x=data.index, y=data[sma_label], mode='lines', name=f'SMA de {sma_window} días'))
                fig.update_layout(title=f"Precio Histórico de {ticker}", xaxis_title="Fecha", yaxis_title="Precio")
                st.plotly_chart(fig, use_container_width=True)

                # Add the rest of your plotting code...

else:
    st.warning("⚠️ Por favor, ingrese un símbolo de ticker válido para comenzar el análisis.")

st.markdown("---")
st.markdown("© 2024 MTaurus. Todos los derechos reservados.")

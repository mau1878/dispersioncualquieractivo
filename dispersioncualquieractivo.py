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
        df.columns = [f"{col[0]} {col[1]}" for col in df.columns]
    else:
        df.columns = [f"{col} {ticker}" for col in df.columns]
    return df

# Función para descargar datos
def download_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            st.warning(f"No hay datos disponibles para el ticker **{ticker}** en el rango de fechas seleccionado.")
            return None
        df = flatten_columns(df, ticker)
        st.write(f"**Columnas para {ticker}:** {df.columns.tolist()}")
        return df
    except Exception as e:
        st.error(f"Error al descargar datos para el ticker **{ticker}**: {e}")
        return None

# Función para analizar la fiabilidad de diferentes MAs
def analyze_ma_reliability(data, price_column, ma_lengths):
    results = []
    for ma_length in ma_lengths:
        ma_label = f'SMA_{ma_length}'
        data[ma_label] = data[price_column].rolling(window=ma_length).mean()
        
        # Detectar cruces (crossovers y crossunders)
        data['Above_MA'] = data[price_column] > data[ma_label]
        data['Crossover'] = data['Above_MA'].diff().fillna(False)
        crossovers = data['Crossover'].sum()  # Total de cruces
        
        # Calcular la frecuencia de reversiones después de un cruce
        reversals = 0
        for i in range(1, len(data) - 1):
            if data['Crossover'].iloc[i]:  # Si hay un cruce
                if data['Above_MA'].iloc[i]:  # Cruce hacia arriba
                    if data[price_column].iloc[i + 1] < data[ma_label].iloc[i + 1]:  # Revierte hacia abajo
                        reversals += 1
                else:  # Cruce hacia abajo
                    if data[price_column].iloc[i + 1] > data[ma_label].iloc[i + 1]:  # Revierte hacia arriba
                        reversals += 1
        
        reversal_rate = reversals / crossovers if crossovers > 0 else 0
        results.append({
            'MA_Length': ma_length,
            'Crossovers': crossovers,
            'Reversals': reversals,
            'Reversal_Rate': reversal_rate
        })
    
    return pd.DataFrame(results)

# Título de la aplicación
st.title("📈 Análisis de SMA y Dispersión de Precios - MTaurus")
st.markdown("### 🚀 Sigue nuestro trabajo en [Twitter](https://twitter.com/MTaurus_ok)")

# Crear pestañas
tab1, tab2 = st.tabs(["Análisis Original", "Análisis de Fiabilidad de MA"])

# Pestaña 1: Análisis Original
with tab1:
    ticker = st.text_input("🖊️ Ingrese el símbolo del ticker", value="GGAL", key="ticker_original").upper()
    
    if ticker:
        sma_window = st.number_input("📊 Ingrese la ventana de SMA (número de días)", min_value=1, value=21, key="sma_original")
        start_date = st.date_input(
            "📅 Seleccione la fecha de inicio",
            value=pd.to_datetime('2000-01-01'),
            min_value=pd.to_datetime('1900-01-01'),
            max_value=pd.to_datetime('today'),
            key="start_original"
        )
        end_date = st.date_input(
            "📅 Seleccione la fecha de fin",
            value=pd.to_datetime('today') + pd.DateOffset(days=1),
            min_value=pd.to_datetime('1900-01-01'),
            max_value=pd.to_datetime('today') + pd.DateOffset(days=1),
            key="end_original"
        )

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        if start_date > end_date:
            st.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
        else:
            close_price_type = st.selectbox("📈 Seleccione el tipo de precio de cierre", ["No ajustado", "Ajustado"], key="price_type_original")
            apply_ratio = st.checkbox("🔄 Ajustar precio por el ratio YPFD.BA/YPF", key="ratio_original")

            data = download_data(ticker, start_date, end_date)

            if data is not None:
                adj_close_col_main = f"Adj Close {ticker}"
                close_col_main = f"Close {ticker}"

                if apply_ratio:
                    st.subheader("🔄 Aplicando ajuste por ratio YPFD.BA/YPF")
                    ypfd_ba_ticker = "YPFD.BA"
                    ypf_ticker = "YPF"
                    ypfd_ba_data = download_data(ypfd_ba_ticker, start_date, end_date)
                    ypf_data = download_data(ypf_ticker, start_date, end_date)

                    if ypfd_ba_data is not None and ypf_data is not None:
                        adj_close_col_ypfd = f"Adj Close {ypfd_ba_ticker}"
                        adj_close_col_ypf = f"Adj Close {ypf_ticker}"
                        close_col_ypfd = f"Close {ypfd_ba_ticker}"
                        close_col_ypf = f"Close {ypf_ticker}"

                        ypfd_price_col = adj_close_col_ypfd if adj_close_col_ypfd in ypfd_ba_data.columns else close_col_ypfd
                        ypf_price_col = adj_close_col_ypf if adj_close_col_ypf in ypf_data.columns else close_col_ypf

                        if ypfd_price_col in ypfd_ba_data.columns and ypf_price_col in ypf_data.columns:
                            ypfd_ba_data = ypfd_ba_data.fillna(method='ffill').fillna(method='bfill')
                            ypf_data = ypf_data.fillna(method='ffill').fillna(method='bfill')
                            ratio = ypfd_ba_data[ypfd_price_col] / ypf_data[ypf_price_col]
                            ratio = ratio.reindex(data.index).fillna(method='ffill').fillna(method='bfill')

                            if adj_close_col_main in data.columns:
                                data['Adj Close Ajustado'] = data[adj_close_col_main] / ratio
                            else:
                                st.warning(f"No se encontró 'Adj Close' para {ticker}. Usando 'Close'.")
                                data['Adj Close Ajustado'] = data[close_col_main] / ratio
                            data['Close Ajustado'] = data[close_col_main] / ratio
                        else:
                            st.error(f"No se encontraron columnas de precio válidas para {ypfd_ba_ticker} o {ypf_ticker}.")
                    else:
                        st.error("No se pudieron descargar los datos necesarios para aplicar el ratio.")
                else:
                    if adj_close_col_main in data.columns:
                        data['Adj Close Original'] = data[adj_close_col_main]
                    else:
                        st.warning(f"No se encontró 'Adj Close' para {ticker}. Usando 'Close'.")
                        data['Adj Close Original'] = data[close_col_main]
                    data['Close Original'] = data[close_col_main]

                price_column = (
                    'Adj Close Ajustado' if (apply_ratio and close_price_type == "Ajustado" and 'Adj Close Ajustado' in data.columns)
                    else 'Close Ajustado' if (apply_ratio and 'Close Ajustado' in data.columns)
                    else 'Adj Close Original' if (not apply_ratio and close_price_type == "Ajustado" and 'Adj Close Original' in data.columns)
                    else 'Close Original' if 'Close Original' in data.columns
                    else close_col_main
                )

                if price_column not in data.columns:
                    st.error(f"La columna seleccionada **{price_column}** no existe en los datos.")
                else:
                    sma_label = f'SMA_{sma_window}'
                    data[sma_label] = data[price_column].rolling(window=sma_window).mean()
                    data['Dispersión'] = data[price_column] - data[sma_label]
                    data['Porcentaje_Dispersión'] = (data['Dispersión'] / data[sma_label]) * 100

                    # Visualización 1: Precio Histórico con SMA
                    st.write("### 📈 Precio Histórico con SMA")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name='Precio de Cierre'))
                    fig.add_trace(go.Scatter(x=data.index, y=data[sma_label], mode='lines', name=f'SMA de {sma_window} días'))
                    fig.add_annotation(text="MTaurus. X: mtaurus_ok", xref="paper", yref="paper", x=0.95, y=0.05, showarrow=False, font=dict(size=14, color="gray"), opacity=0.5)
                    fig.update_layout(
                        title=f"Precio Histórico {'Ajustado' if close_price_type == 'Ajustado' else 'No Ajustado'} de {ticker} con SMA de {sma_window} días",
                        xaxis_title="Fecha", yaxis_title="Precio (USD)", legend_title="Leyenda", template="plotly_dark", hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Visualización 2: Porcentaje de Dispersión Histórico
                    st.write("### 📉 Porcentaje de Dispersión Histórico")
                    fig_dispersion = go.Figure()
                    fig_dispersion.add_trace(go.Scatter(x=data.index, y=data['Porcentaje_Dispersión'], mode='lines', name='Porcentaje de Dispersión'))
                    fig_dispersion.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=0, y1=0, line=dict(color="red", width=2))
                    fig_dispersion.add_annotation(text="MTaurus. X: mtaurus_ok", xref="paper", yref="paper", x=0.95, y=0.05, showarrow=False, font=dict(size=14, color="gray"), opacity=0.5)
                    fig_dispersion.update_layout(
                        title=f"Porcentaje de Dispersión Histórico de {ticker} ({close_price_type})",
                        xaxis_title="Fecha", yaxis_title="Dispersión (%)", legend_title="Leyenda", template="plotly_dark", hovermode="x unified"
                    )
                    st.plotly_chart(fig_dispersion, use_container_width=True)

                    # Visualización 3: Histograma con Seaborn/Matplotlib
                    st.write("### 📊 Histograma de Porcentaje de Dispersión con Percentiles")
                    percentiles = [95, 85, 75, 50, 25, 15, 5]
                    percentile_values = np.percentile(data['Porcentaje_Dispersión'].dropna(), percentiles)
                    plt.figure(figsize=(10, 6))
                    sns.histplot(data['Porcentaje_Dispersión'].dropna(), kde=True, color='blue', bins=100)
                    for percentile, value in zip(percentiles, percentile_values):
                        plt.axvline(value, color='red', linestyle='--')
                        plt.text(value, plt.ylim()[1] * 0.9, f'{percentile}º Percentil', color='red', rotation='vertical', verticalalignment='center', horizontalalignment='right')
                    plt.text(0.95, 0.05, "MTaurus. X: mtaurus_ok", fontsize=14, color='gray', ha='right', va='center', alpha=0.5, transform=plt.gcf().transFigure)
                    plt.title(f'Porcentaje de Dispersión de {ticker} ({close_price_type}) desde SMA de {sma_window} días')
                    plt.xlabel('Dispersión (%)')
                    plt.ylabel('Frecuencia')
                    plt.tight_layout()
                    st.pyplot(plt)
                    plt.clf()

                    # Visualización 4: Histograma Personalizable con Plotly
                    st.write("### 🎨 Personalización del Histograma")
                    num_bins = st.slider("Seleccione el número de bins para el histograma", min_value=10, max_value=100, value=50, key="bins_original")
                    hist_color = st.color_picker("Elija un color para el histograma", value='#1f77b4', key="color_original")
                    st.write("### 📊 Histograma de Porcentaje de Dispersión")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(x=data['Porcentaje_Dispersión'].dropna(), nbinsx=num_bins, marker_color=hist_color, opacity=0.75, name="Histograma"))
                    for percentile, value in zip(percentiles, percentile_values):
                        fig_hist.add_vline(x=value, line=dict(color="red", width=2, dash="dash"), annotation_text=f'{percentile}º Percentil', annotation_position="top", annotation=dict(textangle=-90, font=dict(color="red")))
                    fig_hist.add_annotation(text="MTaurus. X: mtaurus_ok", xref="paper", yref="paper", x=0.95, y=0.05, showarrow=False, font=dict(size=14, color="gray"), opacity=0.5)
                    fig_hist.update_layout(
                        title=f'Histograma del Porcentaje de Dispersión de {ticker} ({close_price_type})',
                        xaxis_title='Dispersión (%)', yaxis_title='Frecuencia', bargap=0.1, template="plotly_dark", hovermode="x unified"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("⚠️ Por favor, ingrese un símbolo de ticker válido para comenzar el análisis.")

# Pestaña 2: Análisis de Fiabilidad de MA
with tab2:
    st.header("Análisis de Fiabilidad de Medias Móviles")
    ticker_ma = st.text_input("🖊️ Ingrese el símbolo del ticker", value="GGAL", key="ticker_ma").upper()
    
    if ticker_ma:
        start_date_ma = st.date_input(
            "📅 Seleccione la fecha de inicio",
            value=pd.to_datetime('2000-01-01'),
            min_value=pd.to_datetime('1900-01-01'),
            max_value=pd.to_datetime('today'),
            key="start_ma"
        )
        end_date_ma = st.date_input(
            "📅 Seleccione la fecha de fin",
            value=pd.to_datetime('today') + pd.DateOffset(days=1),
            min_value=pd.to_datetime('1900-01-01'),
            max_value=pd.to_datetime('today') + pd.DateOffset(days=1),
            key="end_ma"
        )
        min_ma_length = st.number_input("Longitud mínima de MA", min_value=1, value=5, key="min_ma")
        max_ma_length = st.number_input("Longitud máxima de MA", min_value=min_ma_length + 1, value=50, key="max_ma")
        step_ma_length = st.number_input("Paso entre longitudes de MA", min_value=1, value=5, key="step_ma")

        start_date_ma = pd.to_datetime(start_date_ma)
        end_date_ma = pd.to_datetime(end_date_ma)

        if start_date_ma > end_date_ma:
            st.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
        else:
            close_price_type_ma = st.selectbox("📈 Seleccione el tipo de precio de cierre", ["No ajustado", "Ajustado"], key="price_type_ma")
            data_ma = download_data(ticker_ma, start_date_ma, end_date_ma)

            if data_ma is not None:
                adj_close_col_main = f"Adj Close {ticker_ma}"
                close_col_main = f"Close {ticker_ma}"
                price_column_ma = adj_close_col_main if (close_price_type_ma == "Ajustado" and adj_close_col_main in data_ma.columns) else close_col_main

                if price_column_ma not in data_ma.columns:
                    st.error(f"La columna seleccionada **{price_column_ma}** no existe en los datos.")
                else:
                    ma_lengths = range(min_ma_length, max_ma_length + 1, step_ma_length)
                    reliability_df = analyze_ma_reliability(data_ma, price_column_ma, ma_lengths)

                    st.write("### Resultados del Análisis de Fiabilidad")
                    st.dataframe(reliability_df)

                    # Visualización: Tasa de Reversión por Longitud de MA
                    fig_reliability = go.Figure()
                    fig_reliability.add_trace(go.Scatter(
                        x=reliability_df['MA_Length'],
                        y=reliability_df['Reversal_Rate'],
                        mode='lines+markers',
                        name='Tasa de Reversión'
                    ))
                    fig_reliability.add_annotation(text="MTaurus. X: mtaurus_ok", xref="paper", yref="paper", x=0.95, y=0.05, showarrow=False, font=dict(size=14, color="gray"), opacity=0.5)
                    fig_reliability.update_layout(
                        title=f"Tasa de Reversión por Longitud de MA para {ticker_ma}",
                        xaxis_title="Longitud de MA (días)",
                        yaxis_title="Tasa de Reversión",
                        template="plotly_dark",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_reliability, use_container_width=True)

                    # Identificar la MA más "fiable"
                    best_ma = reliability_df.loc[reliability_df['Reversal_Rate'].idxmax()]
                    st.write(f"**MA más fiable**: {int(best_ma['MA_Length'])} días (Tasa de Reversión: {best_ma['Reversal_Rate']:.2%})")
    else:
        st.warning("⚠️ Por favor, ingrese un símbolo de ticker válido para comenzar el análisis.")

# Footer
st.markdown("---")
st.markdown("© 2024 MTaurus. Todos los derechos reservados.")

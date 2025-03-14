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

# Función para analizar SMAs y encontrar la más "confiable"
def analyze_sma_reliability(data, price_column, sma_lengths):
    results = []
    for sma_length in sma_lengths:
        sma_label = f'SMA_{sma_length}'
        data[sma_label] = data[price_column].rolling(window=sma_length).mean()

        # Calcular cruces (crossovers)
        data['Above_SMA'] = data[price_column] > data[sma_label]
        data['Crossover'] = data['Above_SMA'].ne(data['Above_SMA'].shift()).astype(int)
        num_crossovers = data['Crossover'].sum()

        # Calcular reversiones exitosas
        reversals = 0
        for i in range(1, len(data) - 1):
            if data['Crossover'].iloc[i] == 1:
                prev_price = data[price_column].iloc[i-1]
                curr_price = data[price_column].iloc[i]
                next_price = data[price_column].iloc[i+1]
                if (curr_price > prev_price and next_price < curr_price) or \
                   (curr_price < prev_price and next_price > curr_price):
                    reversals += 1
        reversal_rate = reversals / num_crossovers if num_crossovers > 0 else 0

        # Distancia promedio al SMA
        avg_distance = (data[price_column] - data[sma_label]).abs().mean()

        results.append({
            'SMA_Length': sma_length,
            'Crossovers': num_crossovers,
            'Reversal_Rate': reversal_rate,
            'Avg_Distance': avg_distance
        })

    return pd.DataFrame(results)

# Título de la aplicación
st.title("📈 Análisis de SMA y Dispersión de Precios - MTaurus")
st.markdown("### 🚀 Sigue nuestro trabajo en [Twitter](https://twitter.com/MTaurus_ok)")

# Entradas del usuario
ticker = st.text_input("🖊️ Ingrese el símbolo del ticker", value="GGAL").upper()

if ticker:
    # Crear pestañas
    tab1, tab2 = st.tabs(["Análisis Original", "Análisis de SMAs"])

    # Entradas compartidas
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
        # Descargar datos una vez para ambas pestañas
        data = download_data(ticker, start_date, end_date)

        if data is not None:
            # Configuración común
            close_price_type = st.selectbox("📈 Seleccione el tipo de precio de cierre", ["No ajustado", "Ajustado"])
            apply_ratio = st.checkbox("🔄 Ajustar precio por el ratio YPFD.BA/YPF")
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
                if adj_close_col_main in data.columns:
                    data['Adj Close Original'] = data[adj_close_col_main]
                else:
                    st.warning(f"No se encontró 'Adj Close' para {ticker}. Usando 'Close'.")
                    data['Adj Close Original'] = data[close_col_main]
                data['Close Original'] = data[close_col_main]

            # Seleccionar el precio de cierre
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
                # Pestaña 1: Análisis Original
                with tab1:
                    sma_window = st.number_input("📊 Ingrese la ventana de SMA (número de días)", min_value=1, value=21, key="sma_window_tab1")
                    sma_label = f'SMA_{sma_window}'
                    data[sma_label] = data[price_column].rolling(window=sma_window).mean()
                    data['Dispersión'] = data[price_column] - data[sma_label]
                    data['Porcentaje_Dispersión'] = (data['Dispersión'] / data[sma_label]) * 100

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

                    st.write("### 🎨 Personalización del Histograma")
                    num_bins = st.slider("Seleccione el número de bins para el histograma", min_value=10, max_value=100, value=50, key="bins_slider_tab1")
                    hist_color = st.color_picker("Elija un color para el histograma", value='#1f77b4', key="color_picker_tab1")
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

                # Pestaña 2: Análisis de SMAs
                with tab2:
                    st.subheader("🔍 Análisis de Confiabilidad de SMAs")
                    sma_range_min = st.number_input("📏 Mínima longitud de SMA", min_value=5, value=5, key="sma_min_tab2")
                    sma_range_max = st.number_input("📏 Máxima longitud de SMA", min_value=10, value=200, key="sma_max_tab2")
                    sma_step = st.number_input("📏 Paso entre SMAs", min_value=1, value=5, key="sma_step_tab2")

                    sma_lengths = range(sma_range_min, sma_range_max + 1, sma_step)
                    sma_analysis = analyze_sma_reliability(data, price_column, sma_lengths)

                    # Mostrar tabla de resultados
                    st.write("#### Resultados del Análisis")
                    st.dataframe(sma_analysis.style.format({
                        'Reversal_Rate': '{:.2%}',
                        'Avg_Distance': '{:.2f}'
                    }))

                    # Gráfico comparativo
                    fig_sma = go.Figure()
                    fig_sma.add_trace(go.Scatter(x=sma_analysis['SMA_Length'], y=sma_analysis['Crossovers'], mode='lines+markers', name='Número de Cruces'))
                    fig_sma.add_trace(go.Scatter(x=sma_analysis['SMA_Length'], y=sma_analysis['Reversal_Rate'] * 100, mode='lines+markers', name='Tasa de Reversiones (%)', yaxis='y2'))
                    fig_sma.update_layout(
                        title=f"Análisis de SMAs para {ticker} ({close_price_type})",
                        xaxis_title="Longitud de SMA (días)",
                        yaxis_title="Número de Cruces",
                        yaxis2=dict(title="Tasa de Reversiones (%)", overlaying='y', side='right'),
                        legend_title="Métricas",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_sma, use_container_width=True)

                    # Identificar la SMA más confiable
                    sma_analysis['Score'] = sma_analysis['Reversal_Rate'] * (sma_analysis['Crossovers'] / sma_analysis['Crossovers'].max())
                    best_sma = sma_analysis.loc[sma_analysis['Score'].idxmax()]
                    st.write(f"**SMA más confiable:** {int(best_sma['SMA_Length'])} días (Tasa de Reversiones: {best_sma['Reversal_Rate']:.2%}, Cruces: {int(best_sma['Crossovers'])})")

else:
    st.warning("⚠️ Por favor, ingrese un símbolo de ticker válido para comenzar el análisis.")

st.markdown("---")
st.markdown("© 2024 MTaurus. Todos los derechos reservados.")

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Análisis de Medias Móviles y Dispersión de Precios",
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

# Función para calcular diferentes tipos de medias móviles
def calculate_moving_average(data, price_column, ma_type, ma_length):
    if ma_type == "SMA":
        return data[price_column].rolling(window=ma_length).mean()
    elif ma_type == "EMA":
        return data[price_column].ewm(span=ma_length, adjust=False).mean()
    elif ma_type == "WMA":
        weights = np.arange(1, ma_length + 1)
        return data[price_column].rolling(window=ma_length).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    return None

# Nueva función para analizar la viabilidad de las MAs como estrategia de trading
def analyze_ma_trading_potential(data, price_column, ma_lengths, ma_type, look_forward_days):
    results = []
    for ma_length in ma_lengths:
        ma_label = f'{ma_type}_{ma_length}'
        data[ma_label] = calculate_moving_average(data, price_column, ma_type, ma_length)
        
        # Detectar cruces (crossovers y crossunders)
        data['Above_MA'] = data[price_column] > data[ma_label]
        data['Crossover'] = data['Above_MA'].diff().fillna(False)
        
        # Identificar señales de compra (cruce hacia abajo) y venta (cruce hacia arriba)
        buy_signals = 0  # Cruce hacia abajo (precio cae por debajo de la MA)
        sell_signals = 0  # Cruce hacia arriba (precio sube por encima de la MA)
        max_gains = []  # Ganancias máximas después de señales de compra
        max_losses = []  # Pérdidas máximas después de señales de venta
        
        for i in range(len(data) - look_forward_days):
            if data['Crossover'].iloc[i]:
                if data['Above_MA'].iloc[i]:  # Cruce hacia arriba (señal de venta)
                    sell_signals += 1
                    # Calcular la pérdida máxima en los próximos look_forward_days
                    future_prices = data[price_column].iloc[i:i + look_forward_days + 1]
                    initial_price = data[price_column].iloc[i]
                    max_loss = ((future_prices.min() - initial_price) / initial_price) * 100  # En porcentaje
                    max_losses.append(max_loss)
                else:  # Cruce hacia abajo (señal de compra)
                    buy_signals += 1
                    # Calcular la ganancia máxima en los próximos look_forward_days
                    future_prices = data[price_column].iloc[i:i + look_forward_days + 1]
                    initial_price = data[price_column].iloc[i]
                    max_gain = ((future_prices.max() - initial_price) / initial_price) * 100  # En porcentaje
                    max_gains.append(max_gain)
        
        avg_max_gain = np.mean(max_gains) if max_gains else 0
        avg_max_loss = np.mean(max_losses) if max_losses else 0
        
        results.append({
            'MA_Length': ma_length,
            'Buy_Signals': buy_signals,
            'Avg_Max_Gain (%)': avg_max_gain,
            'Sell_Signals': sell_signals,
            'Avg_Max_Loss (%)': avg_max_loss
        })
    
    return pd.DataFrame(results)

# Título de la aplicación
st.title("📈 Análisis de Medias Móviles y Dispersión de Precios - MTaurus")
st.markdown("### 🚀 Sigue nuestro trabajo en [Twitter](https://twitter.com/MTaurus_ok)")

# Crear pestañas
tab1, tab2 = st.tabs(["Análisis Original", "Análisis de Trading con MA"])

# Pestaña 1: Análisis Original (sin cambios)
with tab1:
    ticker = st.text_input("🖊️ Ingrese el símbolo del ticker", value="GGAL", key="ticker_original").upper()
    
    if ticker:
        ma_type = st.selectbox("📊 Seleccione el tipo de media móvil", ["SMA", "EMA", "WMA"], key="ma_type_original")
        ma_window = st.number_input("📊 Ingrese la ventana de la media móvil (número de días)", min_value=1, value=21, key="ma_window_original")
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
                    ma_label = f'{ma_type}_{ma_window}'
                    data[ma_label] = calculate_moving_average(data, price_column, ma_type, ma_window)
                    data['Dispersión'] = data[price_column] - data[ma_label]
                    data['Porcentaje_Dispersión'] = (data['Dispersión'] / data[ma_label]) * 100

                    # Visualización 1: Precio Histórico con MA
                    st.write(f"### 📈 Precio Histórico con {ma_type}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name='Precio de Cierre'))
                    fig.add_trace(go.Scatter(x=data.index, y=data[ma_label], mode='lines', name=f'{ma_type} de {ma_window} días'))
                    fig.add_annotation(text="MTaurus. X: mtaurus_ok", xref="paper", yref="paper", x=0.95, y=0.05, showarrow=False, font=dict(size=14, color="gray"), opacity=0.5)
                    fig.update_layout(
                        title=f"Precio Histórico {'Ajustado' if close_price_type == 'Ajustado' else 'No Ajustado'} de {ticker} con {ma_type} de {ma_window} días",
                        xaxis_title="Fecha", yaxis_title="Precio (USD)", legend_title="Leyenda", template="plotly_dark", hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Visualización 2: Porcentaje de Dispersión Histórico
                    st.write("### 📉 Porcentaje de Dispersión Histórico")

                    # Verificar datos antes de graficar
                    if data['Porcentaje_Dispersión'].dropna().empty:
                        st.error("No hay datos válidos de dispersión para graficar.")
                    else:
                        fig_dispersion = go.Figure()
                        fig_dispersion.add_trace(go.Scatter(
                            x=data.index, 
                            y=data['Porcentaje_Dispersión'], 
                            mode='lines', 
                            name='Porcentaje de Dispersión',
                            line=dict(color='lightgrey')
                        ))

                        # Línea de promedio histórico (azul claro)
                        historical_mean = data['Porcentaje_Dispersión'].mean()
                        if not pd.isna(historical_mean):
                            fig_dispersion.add_shape(
                                type="line", 
                                x0=data.index.min(), 
                                x1=data.index.max(), 
                                y0=historical_mean, 
                                y1=historical_mean,
                                line=dict(color="lightblue", width=1, dash="dash"),
                            )
                            # Dummy trace para la leyenda
                            fig_dispersion.add_trace(go.Scatter(
                                x=[None], y=[None], mode='lines',
                                line=dict(color="lightblue", width=1, dash="dash"),
                                name=f"Promedio: {historical_mean:.2f}%",
                                showlegend=True,
                                opacity=0
                            ))
                            fig_dispersion.add_annotation(
                                x=data.index.max(), 
                                y=historical_mean, 
                                text=f"Promedio: {historical_mean:.2f}%",
                                showarrow=True, 
                                arrowhead=1, 
                                ax=20, 
                                ay=-20, 
                                font=dict(color="lightblue")
                            )
                        else:
                            st.warning("No se pudo calcular el promedio histórico debido a datos insuficientes.")

                        # Percentiles dinámicos
                        lower_percentile = st.slider("Seleccione el percentil inferior", min_value=1, max_value=49, value=5, key="lower_percentile")
                        upper_percentile = st.slider("Seleccione el percentil superior", min_value=51, max_value=99, value=95, key="upper_percentile")

                        dispersion_data = data['Porcentaje_Dispersión'].dropna()
                        lower_value = np.percentile(dispersion_data, lower_percentile)
                        upper_value = np.percentile(dispersion_data, upper_percentile)

                        # Línea de percentil inferior (rojo)
                        fig_dispersion.add_shape(
                            type="line", 
                            x0=data.index.min(), 
                            x1=data.index.max(), 
                            y0=lower_value, 
                            y1=lower_value,
                            line=dict(color="red", width=1, dash="dash"),
                        )
                        # Dummy trace para la leyenda
                        fig_dispersion.add_trace(go.Scatter(
                            x=[None], y=[None], mode='lines',
                            line=dict(color="red", width=1, dash="dash"),
                            name=f"P{lower_percentile}: {lower_value:.2f}%",
                            showlegend=True,
                            opacity=0
                        ))
                        fig_dispersion.add_annotation(
                            x=data.index.max(), 
                            y=lower_value, 
                            text=f"P{lower_percentile}: {lower_value:.2f}%",
                            showarrow=True, 
                            arrowhead=1, 
                            ax=20, 
                            ay=20, 
                            font=dict(color="red")
                        )

                        # Línea de percentil superior (verde)
                        fig_dispersion.add_shape(
                            type="line", 
                            x0=data.index.min(), 
                            x1=data.index.max(), 
                            y0=upper_value, 
                            y1=upper_value,
                            line=dict(color="green", width=1, dash="dash"),
                        )
                        # Dummy trace para la leyenda
                        fig_dispersion.add_trace(go.Scatter(
                            x=[None], y=[None], mode='lines',
                            line=dict(color="green", width=1, dash="dash"),
                            name=f"P{upper_percentile}: {upper_value:.2f}%",
                            showlegend=True,
                            opacity=0
                        ))
                        fig_dispersion.add_annotation(
                            x=data.index.max(), 
                            y=upper_value, 
                            text=f"P{upper_percentile}: {upper_value:.2f}%",
                            showarrow=True, 
                            arrowhead=1, 
                            ax=20, 
                            ay=-20, 
                            font=dict(color="green")
                        )

                        # Línea cero (como en el original)
                        fig_dispersion.add_shape(
                            type="line", 
                            x0=data.index.min(), 
                            x1=data.index.max(), 
                            y0=0, 
                            y1=0, 
                            line=dict(color="red", width=2)
                        )
                        # Dummy trace para la leyenda (línea cero)
                        fig_dispersion.add_trace(go.Scatter(
                            x=[None], y=[None], mode='lines',
                            line=dict(color="red", width=2),
                            name="Línea Cero",
                            showlegend=True,
                            opacity=0
                        ))

                        # Anotación de MTaurus
                        fig_dispersion.add_annotation(
                            text="MTaurus. X: mtaurus_ok", 
                            xref="paper", 
                            yref="paper", 
                            x=0.95, 
                            y=0.05,
                            showarrow=False, 
                            font=dict(size=14, color="gray"), 
                            opacity=0.5
                        )

                        # Configuración del layout
                        fig_dispersion.update_layout(
                            title=f"Porcentaje de Dispersión Histórico de {ticker} ({close_price_type})",
                            xaxis_title="Fecha", 
                            yaxis_title="Dispersión (%)", 
                            legend_title="Leyenda",
                            template="plotly_dark", 
                            hovermode="x unified",
                            showlegend=True
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
                    plt.title(f'Porcentaje de Dispersión de {ticker} ({close_price_type}) desde {ma_type} de {ma_window} días')
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

# Pestaña 2: Análisis de Trading con MA (modificado)
with tab2:
    st.header("Análisis de Trading con Medias Móviles")
    
    st.markdown("""
    ### ¿Qué hace esta pestaña?
    Esta herramienta evalúa medias móviles (MA) para identificar las más útiles en una estrategia de trading. Analizamos:
    - **Señales de Compra**: Cuando el precio cruza hacia abajo de la MA (potencial oportunidad de compra).
    - **Señales de Venta**: Cuando el precio cruza hacia arriba de la MA (potencial oportunidad de venta).
    Para cada señal, calculamos:
    - La **ganancia máxima promedio** después de una señal de compra (en los próximos N días).
    - La **pérdida máxima promedio** después de una señal de venta (en los próximos N días).
    Esto te ayuda a elegir una MA que ofrezca buenas oportunidades de ganancia con un riesgo controlado.
    """)

    ticker_ma = st.text_input("🖊️ Ingrese el símbolo del ticker", value="GGAL", key="ticker_ma").upper()
    
    if ticker_ma:
        ma_type_ma = st.selectbox("📊 Seleccione el tipo de media móvil", ["SMA", "EMA", "WMA"], key="ma_type_ma")
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
        look_forward_days = st.number_input("Días de proyección (N días después de la señal)", min_value=1, value=5, key="look_forward_days")
        close_price_type_ma = st.selectbox("📈 Seleccione el tipo de precio de cierre", ["No ajustado", "Ajustado"], key="price_type_ma")
        apply_ratio_ma = st.checkbox("🔄 Ajustar precio por el ratio YPFD.BA/YPF", key="ratio_ma")

        start_date_ma = pd.to_datetime(start_date_ma)
        end_date_ma = pd.to_datetime(end_date_ma)

        if start_date_ma > end_date_ma:
            st.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
        else:
            if st.button("Confirmar Análisis", key="confirm_ma"):
                data_ma = download_data(ticker_ma, start_date_ma, end_date_ma)

                if data_ma is not None:
                    adj_close_col_main = f"Adj Close {ticker_ma}"
                    close_col_main = f"Close {ticker_ma}"

                    if apply_ratio_ma:
                        st.subheader("🔄 Aplicando ajuste por ratio YPFD.BA/YPF")
                        ypfd_ba_ticker = "YPFD.BA"
                        ypf_ticker = "YPF"
                        ypfd_ba_data = download_data(ypfd_ba_ticker, start_date_ma, end_date_ma)
                        ypf_data = download_data(ypf_ticker, start_date_ma, end_date_ma)

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
                                ratio = ratio.reindex(data_ma.index).fillna(method='ffill').fillna(method='bfill')

                                if adj_close_col_main in data_ma.columns:
                                    data_ma['Adj Close Ajustado'] = data_ma[adj_close_col_main] / ratio
                                else:
                                    st.warning(f"No se encontró 'Adj Close' para {ticker_ma}. Usando 'Close'.")
                                    data_ma['Adj Close Ajustado'] = data_ma[close_col_main] / ratio
                                data_ma['Close Ajustado'] = data_ma[close_col_main] / ratio
                            else:
                                st.error(f"No se encontraron columnas de precio válidas para {ypfd_ba_ticker} o {ypf_ticker}.")
                        else:
                            st.error("No se pudieron descargar los datos necesarios para aplicar el ratio.")
                    else:
                        if adj_close_col_main in data_ma.columns:
                            data_ma['Adj Close Original'] = data_ma[adj_close_col_main]
                        else:
                            st.warning(f"No se encontró 'Adj Close' para {ticker_ma}. Usando 'Close'.")
                            data_ma['Adj Close Original'] = data_ma[close_col_main]
                        data_ma['Close Original'] = data_ma[close_col_main]

                    price_column_ma = (
                        'Adj Close Ajustado' if (apply_ratio_ma and close_price_type_ma == "Ajustado" and 'Adj Close Ajustado' in data_ma.columns)
                        else 'Close Ajustado' if (apply_ratio_ma and 'Close Ajustado' in data_ma.columns)
                        else 'Adj Close Original' if (not apply_ratio_ma and close_price_type_ma == "Ajustado" and 'Adj Close Original' in data_ma.columns)
                        else 'Close Original' if 'Close Original' in data_ma.columns
                        else close_col_main
                    )

                    if price_column_ma not in data_ma.columns:
                        st.error(f"La columna seleccionada **{price_column_ma}** no existe en los datos.")
                    else:
                        ma_lengths = range(min_ma_length, max_ma_length + 1, step_ma_length)
                        trading_df = analyze_ma_trading_potential(data_ma, price_column_ma, ma_lengths, ma_type_ma, look_forward_days)

                        st.write("### Resultados del Análisis de Trading")
                        st.markdown("""
                        Aquí tienes una tabla con los resultados:
                        - **MA_Length**: El número de días de la media móvil.
                        - **Buy_Signals**: Cuántas veces el precio cruzó hacia abajo de la MA (señal de compra).
                        - **Avg_Max_Gain (%)**: Ganancia máxima promedio después de una señal de compra (en los próximos N días).
                        - **Sell_Signals**: Cuántas veces el precio cruzó hacia arriba de la MA (señal de venta).
                        - **Avg_Max_Loss (%)**: Pérdida máxima promedio después de una señal de venta (en los próximos N días).
                        """)
                        st.dataframe(trading_df)

                        # Visualización: Ganancia y Pérdida Máxima Promedio por Longitud de MA
                        fig_trading = go.Figure()
                        fig_trading.add_trace(go.Scatter(
                            x=trading_df['MA_Length'],
                            y=trading_df['Avg_Max_Gain (%)'],
                            mode='lines+markers',
                            name='Ganancia Máx. Promedio (%)',
                            line=dict(color='green')
                        ))
                        fig_trading.add_trace(go.Scatter(
                            x=trading_df['MA_Length'],
                            y=trading_df['Avg_Max_Loss (%)'],
                            mode='lines+markers',
                            name='Pérdida Máx. Promedio (%)',
                            line=dict(color='red')
                        ))
                        fig_trading.add_annotation(
                            text="MTaurus. X: mtaurus_ok", 
                            xref="paper", 
                            yref="paper", 
                            x=0.95, 
                            y=0.05, 
                            showarrow=False, 
                            font=dict(size=14, color="gray"), 
                            opacity=0.5
                        )
                        fig_trading.update_layout(
                            title=f"Ganancia y Pérdida Máxima Promedio por Longitud de {ma_type_ma} para {ticker_ma}",
                            xaxis_title="Longitud de MA (días)",
                            yaxis_title="Porcentaje (%)",
                            template="plotly_dark",
                            hovermode="x unified",
                            showlegend=True
                        )
                        st.plotly_chart(fig_trading, use_container_width=True)

                        # Identificar la MA más "viable" para trading
                        # Podríamos usar un criterio simple: mayor ganancia promedio con menor pérdida promedio
                        trading_df['Gain_Loss_Ratio'] = trading_df['Avg_Max_Gain (%)'] / abs(trading_df['Avg_Max_Loss (%)']).replace(0, np.nan)
                        best_ma = trading_df.loc[trading_df['Gain_Loss_Ratio'].idxmax()]
                        st.markdown(f"""
                        ### ¿Cuál es la mejor {ma_type_ma} para trading?
                        Basado en los datos, la {ma_type_ma} de **{int(best_ma['MA_Length'])} días** parece ser la más viable para {ticker_ma}. 
                        - **Ganancia Máxima Promedio**: {best_ma['Avg_Max_Gain (%)']:.2f}% después de una señal de compra.
                        - **Pérdida Máxima Promedio**: {best_ma['Avg_Max_Loss (%)']:.2f}% después de una señal de venta.
                        Esto sugiere que podrías comprar cuando el precio cruza hacia abajo de esta MA y esperar una ganancia promedio de {best_ma['Avg_Max_Gain (%)']:.2f}% en los próximos {look_forward_days} días, mientras que las señales de venta tienen un riesgo promedio de {best_ma['Avg_Max_Loss (%)']:.2f}%.
                        """)
    else:
        st.warning("⚠️ Por favor, ingrese un símbolo de ticker válido para comenzar el análisis.")

# Footer
st.markdown("---")
st.markdown("© 2024 MTaurus. Todos los derechos reservados.")

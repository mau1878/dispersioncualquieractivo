import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set Matplotlib backend to 'agg' to avoid rendering issues in Streamlit
plt.switch_backend('agg')

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
    return df

# Función para descargar y comprimir datos
def download_data(ticker, start, end, compression='Daily'):
    try:
        # Descargar datos diarios de yfinance
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            st.warning(f"No hay datos disponibles para el ticker **{ticker}** en el rango de fechas seleccionado.")
            return None
        
        # Aplanar columnas
        df = flatten_columns(df, ticker)

        # Mapear la compresión seleccionada a la regla de pandas
        if compression == 'Weekly':
            rule = 'W'  # Semanal
            df = df.resample(rule).agg({
                f'Open {ticker}': 'first',
                f'High {ticker}': 'max',
                f'Low {ticker}': 'min',
                f'Close {ticker}': 'last',
                f'Volume {ticker}': 'sum'
            })
        elif compression == 'Monthly':
            rule = 'M'  # Mensual
            df = df.resample(rule).agg({
                f'Open {ticker}': 'first',
                f'High {ticker}': 'max',
                f'Low {ticker}': 'min',
                f'Close {ticker}': 'last',
                f'Volume {ticker}': 'sum'
            })
        else:
            rule = 'D'  # Diario (sin cambios)
        
        st.write(f"**Columnas para {ticker} ({compression}):** {df.columns.tolist()}")
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

# Función para analizar la estrategia basada en percentiles de dispersión
def analyze_ma_percentile_strategy(data, price_column, ma_lengths, ma_type, look_forward_days, low_percentile, high_percentile):
    if len(data) < look_forward_days + 1:
        st.error(f"El conjunto de datos es demasiado corto para analizar con {look_forward_days} días de proyección. Se necesitan al menos {look_forward_days + 1} días de datos.")
        return pd.DataFrame()

    results = []
    for ma_length in ma_lengths:
        ma_label = f'{ma_type}_{ma_length}'
        data[ma_label] = calculate_moving_average(data, price_column, ma_type, ma_length)
        
        # Calcular la dispersión
        data['Dispersion'] = (data[price_column] - data[ma_label]) / data[ma_label] * 100
        
        # Determinar los percentiles de dispersión históricos
        dispersion_data = data['Dispersion'].dropna()
        if len(dispersion_data) < 10:  # Asegurarse de que haya suficientes datos para calcular percentiles
            st.warning(f"No hay suficientes datos de dispersión para {ma_type} de longitud {ma_length}.")
            continue
        low_threshold = np.percentile(dispersion_data, low_percentile)
        high_threshold = np.percentile(dispersion_data, high_percentile)
        
        # Identificar señales de compra y venta basadas en percentiles
        buy_signals = 0
        sell_signals = 0
        buy_successes = 0
        sell_successes = 0
        buy_gains = []
        sell_gains = []
        
        for i in range(len(data) - look_forward_days):
            current_dispersion = data['Dispersion'].iloc[i]
            initial_price = data[price_column].iloc[i]
            future_prices = data[price_column].iloc[i:i + look_forward_days + 1]
            
            # Verificar que los datos sean válidos
            if pd.isna(current_dispersion) or pd.isna(initial_price) or initial_price == 0 or future_prices.isna().any():
                continue
            
            # Señal de compra: dispersión por debajo del percentil bajo
            if current_dispersion <= low_threshold:
                buy_signals += 1
                future_max = future_prices.max()
                gain = (future_max - initial_price) / initial_price * 100
                buy_gains.append(gain)
                if future_max > initial_price:  # Éxito si el precio sube
                    buy_successes += 1
            
            # Señal de venta: dispersión por encima del percentil alto
            elif current_dispersion >= high_threshold:
                sell_signals += 1
                future_min = future_prices.min()
                loss = (future_min - initial_price) / initial_price * 100
                sell_gains.append(loss)
                if future_min < initial_price:  # Éxito si el precio baja
                    sell_successes += 1
        
        if buy_signals == 0 and sell_signals == 0:
            st.warning(f"No se encontraron señales de compra o venta para {ma_type} de longitud {ma_length} con los percentiles seleccionados.")
            continue
        
        buy_success_rate = (buy_successes / buy_signals * 100) if buy_signals > 0 else 0
        sell_success_rate = (sell_successes / sell_signals * 100) if sell_signals > 0 else 0
        avg_buy_gain = np.mean(buy_gains) if buy_gains else 0
        avg_sell_gain = np.mean(sell_gains) if sell_gains else 0
        
        results.append({
            'MA_Length': ma_length,
            'Buy_Signals': buy_signals,
            'Buy_Success_Rate (%)': buy_success_rate,
            'Avg_Buy_Gain (%)': avg_buy_gain,
            'Sell_Signals': sell_signals,
            'Sell_Success_Rate (%)': sell_success_rate,
            'Avg_Sell_Gain (%)': avg_sell_gain
        })
    
    return pd.DataFrame(results)

# Título de la aplicación
st.title("📈 Análisis de Medias Móviles y Dispersión de Precios - MTaurus")
st.markdown("### 🚀 Sigue nuestro trabajo en [Twitter](https://twitter.com/MTaurus_ok)")

# Crear pestañas
tab1, tab2 = st.tabs(["Análisis Original", "Análisis de Trading con Percentiles de Dispersión"])

# Pestaña 1: Análisis Original (con opción de compresión y sin Adj Close)
with tab1:
    ticker = st.text_input("🖊️ Ingrese el símbolo del ticker", value="AAPL", key="ticker_original").upper()
    
    if ticker:
        ma_type = st.selectbox("📊 Seleccione el tipo de media móvil", ["SMA", "EMA", "WMA"], key="ma_type_original")
        ma_window = st.number_input("📊 Ingrese la ventana de la media móvil (número de períodos)", min_value=1, value=21, key="ma_window_original")
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
        compression = st.selectbox("📅 Seleccione la compresión de datos", ["Daily", "Weekly", "Monthly"], key="compression_original")

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        if start_date > end_date:
            st.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
        else:
            apply_ratio = st.checkbox("🔄 Ajustar precio por el ratio YPFD.BA/YPF", key="ratio_original")

            data = download_data(ticker, start_date, end_date, compression=compression)

            if data is not None:
                close_col_main = f"Close {ticker}"

                if apply_ratio:
                    st.subheader("🔄 Aplicando ajuste por ratio YPFD.BA/YPF")
                    ypfd_ba_ticker = "YPFD.BA"
                    ypf_ticker = "YPF"
                    ypfd_ba_data = download_data(ypfd_ba_ticker, start_date, end_date, compression=compression)
                    ypf_data = download_data(ypf_ticker, start_date, end_date, compression=compression)

                    if ypfd_ba_data is not None and ypf_data is not None:
                        close_col_ypfd = f"Close {ypfd_ba_ticker}"
                        close_col_ypf = f"Close {ypf_ticker}"

                        if close_col_ypfd in ypfd_ba_data.columns and close_col_ypf in ypf_data.columns:
                            ypfd_ba_data = ypfd_ba_data.fillna(method='ffill').fillna(method='bfill')
                            ypf_data = ypf_data.fillna(method='ffill').fillna(method='bfill')
                            ratio = ypfd_ba_data[close_col_ypfd] / ypf_data[close_col_ypf]
                            ratio = ratio.reindex(data.index).fillna(method='ffill').fillna(method='bfill')

                            data['Close Ajustado'] = data[close_col_main] / ratio
                        else:
                            st.error(f"No se encontraron columnas de precio válidas para {ypfd_ba_ticker} o {ypf_ticker}.")
                    else:
                        st.error("No se pudieron descargar los datos necesarios para aplicar el ratio.")
                else:
                    data['Close Original'] = data[close_col_main]

                price_column = (
                    'Close Ajustado' if (apply_ratio and 'Close Ajustado' in data.columns)
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
                    st.write(f"### 📈 Precio Histórico con {ma_type} ({compression})")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name='Precio de Cierre'))
                    fig.add_trace(go.Scatter(x=data.index, y=data[ma_label], mode='lines', name=f'{ma_type} de {ma_window} períodos'))
                    fig.add_annotation(text="MTaurus. X: mtaurus_ok", xref="paper", yref="paper", x=0.95, y=0.05, showarrow=False, font=dict(size=14, color="gray"), opacity=0.5)
                    fig.update_layout(
                        title=f"Precio Histórico de {ticker} con {ma_type} de {ma_window} períodos ({compression})",
                        xaxis_title="Fecha", yaxis_title="Precio (USD)", legend_title="Leyenda", template="plotly_dark", hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Visualización 2: Porcentaje de Dispersión Histórico
                    st.write(f"### 📉 Porcentaje de Dispersión Histórico ({compression})")

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
                            title=f"Porcentaje de Dispersión Histórico de {ticker} ({compression})",
                            xaxis_title="Fecha", 
                            yaxis_title="Dispersión (%)", 
                            legend_title="Leyenda",
                            template="plotly_dark", 
                            hovermode="x unified",
                            showlegend=True
                        )

                        st.plotly_chart(fig_dispersion, use_container_width=True)

                    # Dentro de Tab 1, reemplazar Visualización 3 y Visualización 4 con lo siguiente:
                    
                    # Visualización 3: Histograma con Seaborn/Matplotlib
                    st.write(f"### 📊 Histograma de Porcentaje de Dispersión con Percentiles ({compression})")
                    percentiles = [95, 85, 75, 50, 25, 15, 5]
                    percentile_values = np.percentile(data['Porcentaje_Dispersión'].dropna(), percentiles)
                    
                    # Input para fechas específicas
                    st.write("#### Seleccionar fechas específicas para destacar en el histograma")
                    num_dates = st.number_input("Número de fechas a destacar", min_value=0, max_value=10, value=0, key="num_dates_hist")
                    selected_dates = []
                    dispersion_values = []
                    if num_dates > 0:
                        for i in range(num_dates):
                            date = st.date_input(
                                f"Seleccione la fecha {i+1}",
                                value=data.index[-1],
                                min_value=data.index[0],
                                max_value=data.index[-1],
                                key=f"hist_date_{i}"
                            )
                            date = pd.to_datetime(date)
                            if date in data.index:
                                selected_dates.append(date)
                                disp_value = data.loc[date, 'Porcentaje_Dispersión']
                                if not pd.isna(disp_value):
                                    dispersion_values.append(disp_value)
                                else:
                                    st.warning(f"No hay datos de dispersión para la fecha {date.strftime('%Y-%m-%d')}.")
                            else:
                                st.warning(f"La fecha {date.strftime('%Y-%m-%d')} no está en el rango de datos.")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(data['Porcentaje_Dispersión'].dropna(), kde=True, color='blue', bins=100, ax=ax)
                    for percentile, value in zip(percentiles, percentile_values):
                        ax.axvline(value, color='red', linestyle='--')
                        ax.text(value, ax.get_ylim()[1] * 0.9, f'{percentile}º Percentil', color='red', rotation='vertical', verticalalignment='center', horizontalalignment='right')
                    # Destacar fechas seleccionadas
                    for date, disp_value in zip(selected_dates, dispersion_values):
                        ax.axvline(disp_value, color='green', linestyle='-', alpha=0.5)
                        ax.text(disp_value, ax.get_ylim()[1] * 0.95, f"{date.strftime('%Y-%m-%d')}\n{disp_value:.2f}%", 
                                color='green', rotation='vertical', verticalalignment='center', horizontalalignment='left')
                    ax.text(0.95, 0.05, "MTaurus. X: mtaurus_ok", fontsize=14, color='gray', ha='right', va='center', alpha=0.5, transform=fig.transFigure)
                    ax.set_title(f'Porcentaje de Dispersión de {ticker} ({compression}) desde {ma_type} de {ma_window} períodos')
                    ax.set_xlabel('Dispersión (%)')
                    ax.set_ylabel('Frecuencia')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Visualización 4: Histograma Personalizable con Plotly
                    st.write(f"### 🎨 Personalización del Histograma ({compression})")
                    num_bins = st.slider("Seleccione el número de bins para el histograma", min_value=10, max_value=100, value=50, key="bins_original")
                    hist_color = st.color_picker("Elija un color para el histograma", value='#1f77b4', key="color_original")
                    st.write(f"### 📊 Histograma de Porcentaje de Dispersión ({compression})")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(x=data['Porcentaje_Dispersión'].dropna(), nbinsx=num_bins, marker_color=hist_color, opacity=0.75, name="Histograma"))
                    for percentile, value in zip(percentiles, percentile_values):
                        fig_hist.add_vline(x=value, line=dict(color="red", width=2, dash="dash"), 
                                           annotation_text=f'{percentile}º Percentil', annotation_position="top", 
                                           annotation=dict(textangle=-90, font=dict(color="red")))
                    # Destacar fechas seleccionadas
                    for date, disp_value in zip(selected_dates, dispersion_values):
                        fig_hist.add_vline(x=disp_value, line=dict(color="green", width=2, dash="solid"), 
                                           annotation_text=f"{date.strftime('%Y-%m-%d')}\n{disp_value:.2f}%", 
                                           annotation_position="top", annotation=dict(textangle=-90, font=dict(color="green")))
                    fig_hist.add_annotation(text="MTaurus. X: mtaurus_ok", xref="paper", yref="paper", x=0.95, y=0.05, 
                                            showarrow=False, font=dict(size=14, color="gray"), opacity=0.5)
                    fig_hist.update_layout(
                        title=f'Histograma del Porcentaje de Dispersión de {ticker} ({compression})',
                        xaxis_title='Dispersión (%)', yaxis_title='Frecuencia', bargap=0.1, 
                        template="plotly_dark", hovermode="x unified"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("⚠️ Por favor, ingrese un símbolo de ticker válido para comenzar el análisis.")

# Pestaña 2: Análisis de Trading con Percentiles de Dispersión (con opción de compresión y sin Adj Close)
with tab2:
    st.header("Análisis de Trading con Percentiles de Dispersión")
    
    st.markdown("""
    ### ¿Qué hace esta pestaña?
    Esta herramienta evalúa medias móviles (MA) para una estrategia de trading basada en la dispersión del precio respecto a la MA. Analizamos:
    - **Señales de Compra**: Cuando la dispersión (diferencia porcentual entre el precio y la MA) cae por debajo de un percentil bajo (e.g., 5º percentil), indicando que el precio está inusualmente bajo.
    - **Señales de Venta**: Cuando la dispersión sube por encima de un percentil alto (e.g., 95º percentil), indicando que el precio está inusualmente alto.
    Para cada señal, calculamos:
    - La **tasa de éxito** de las señales de compra (qué tan seguido el precio sube después de una señal de compra) y de venta (qué tan seguido el precio baja después de una señal de venta).
    - La **ganancia promedio** después de una señal de compra y la **pérdida promedio** después de una señal de venta (en los próximos N períodos).
    Esto te ayuda a elegir una MA que ofrezca señales confiables para comprar y vender basadas en extremos de dispersión.
    """)

    ticker_ma = st.text_input("🖊️ Ingrese el símbolo del ticker", value="AAPL", key="ticker_ma").upper()
    
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
        look_forward_days = st.number_input("Períodos de proyección (N períodos después de la señal)", min_value=1, value=5, key="look_forward_days")
        low_percentile = st.slider("Percentil bajo para señales de compra", min_value=1, max_value=49, value=5, key="low_percentile_ma")
        high_percentile = st.slider("Percentil alto para señales de venta", min_value=51, max_value=99, value=95, key="high_percentile_ma")
        compression_ma = st.selectbox("📅 Seleccione la compresión de datos", ["Daily", "Weekly", "Monthly"], key="compression_ma")
        apply_ratio_ma = st.checkbox("🔄 Ajustar precio por el ratio YPFD.BA/YPF", key="ratio_ma")

        start_date_ma = pd.to_datetime(start_date_ma)
        end_date_ma = pd.to_datetime(end_date_ma)

        if start_date_ma > end_date_ma:
            st.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
        else:
            if st.button("Confirmar Análisis", key="confirm_ma"):
                data_ma = download_data(ticker_ma, start_date_ma, end_date_ma, compression=compression_ma)

                if data_ma is not None:
                    close_col_main = f"Close {ticker_ma}"

                    if apply_ratio_ma:
                        st.subheader("🔄 Aplicando ajuste por ratio YPFD.BA/YPF")
                        ypfd_ba_ticker = "YPFD.BA"
                        ypf_ticker = "YPF"
                        ypfd_ba_data = download_data(ypfd_ba_ticker, start_date_ma, end_date_ma, compression=compression_ma)
                        ypf_data = download_data(ypf_ticker, start_date_ma, end_date_ma, compression=compression_ma)

                        if ypfd_ba_data is not None and ypf_data is not None:
                            close_col_ypfd = f"Close {ypfd_ba_ticker}"
                            close_col_ypf = f"Close {ypf_ticker}"

                            if close_col_ypfd in ypfd_ba_data.columns and close_col_ypf in ypf_data.columns:
                                ypfd_ba_data = ypfd_ba_data.fillna(method='ffill').fillna(method='bfill')
                                ypf_data = ypf_data.fillna(method='ffill').fillna(method='bfill')
                                ratio = ypfd_ba_data[close_col_ypfd] / ypf_data[close_col_ypf]
                                ratio = ratio.reindex(data_ma.index).fillna(method='ffill').fillna(method='bfill')

                                data_ma['Close Ajustado'] = data_ma[close_col_main] / ratio
                            else:
                                st.error(f"No se encontraron columnas de precio válidas para {ypfd_ba_ticker} o {ypf_ticker}.")
                        else:
                            st.error("No se pudieron descargar los datos necesarios para aplicar el ratio.")
                    else:
                        data_ma['Close Original'] = data_ma[close_col_main]

                    price_column_ma = (
                        'Close Ajustado' if (apply_ratio_ma and 'Close Ajustado' in data_ma.columns)
                        else 'Close Original' if 'Close Original' in data_ma.columns
                        else close_col_main
                    )

                    if price_column_ma not in data_ma.columns:
                        st.error(f"La columna seleccionada **{price_column_ma}** no existe en los datos.")
                    else:
                        ma_lengths = range(min_ma_length, max_ma_length + 1, step_ma_length)
                        trading_df = analyze_ma_percentile_strategy(
                            data_ma, price_column_ma, ma_lengths, ma_type_ma, look_forward_days, low_percentile, high_percentile
                        )

                        if not trading_df.empty:
                            st.write("### Resultados del Análisis de Trading")
                            st.markdown("""
                            Aquí tienes una tabla con los resultados:
                            - **MA_Length**: El número de períodos de la media móvil.
                            - **Buy_Signals**: Cuántas veces la dispersión cayó por debajo del percentil bajo (señal de compra).
                            - **Buy_Success_Rate (%)**: Porcentaje de señales de compra que resultaron en un aumento del precio.
                            - **Avg_Buy_Gain (%)**: Ganancia promedio después de una señal de compra (en los próximos N períodos).
                            - **Sell_Signals**: Cuántas veces la dispersión subió por encima del percentil alto (señal de venta).
                            - **Sell_Success_Rate (%)**: Porcentaje de señales de venta que resultaron en una caída del precio.
                            - **Avg_Sell_Gain (%)**: Pérdida promedio después de una señal de venta (en los próximos N períodos).
                            """)
                            st.dataframe(trading_df)

                            # Visualización: Tasa de Éxito y Ganancia Promedio por Longitud de MA
                            fig_trading = go.Figure()
                            fig_trading.add_trace(go.Scatter(
                                x=trading_df['MA_Length'],
                                y=trading_df['Buy_Success_Rate (%)'],
                                mode='lines+markers',
                                name='Tasa de Éxito Compra (%)',
                                line=dict(color='green')
                            ))
                            fig_trading.add_trace(go.Scatter(
                                x=trading_df['MA_Length'],
                                y=trading_df['Sell_Success_Rate (%)'],
                                mode='lines+markers',
                                name='Tasa de Éxito Venta (%)',
                                line=dict(color='red')
                            ))
                            fig_trading.add_trace(go.Scatter(
                                x=trading_df['MA_Length'],
                                y=trading_df['Avg_Buy_Gain (%)'],
                                mode='lines+markers',
                                name='Ganancia Promedio Compra (%)',
                                line=dict(color='blue', dash='dash')
                            ))
                            fig_trading.add_trace(go.Scatter(
                                x=trading_df['MA_Length'],
                                y=trading_df['Avg_Sell_Gain (%)'],
                                mode='lines+markers',
                                name='Pérdida Promedio Venta (%)',
                                line=dict(color='orange', dash='dash')
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
                                title=f"Tasa de Éxito y Ganancia/Pérdida Promedio por Longitud de {ma_type_ma} para {ticker_ma} ({compression_ma})",
                                xaxis_title="Longitud de MA (períodos)",
                                yaxis_title="Porcentaje (%)",
                                template="plotly_dark",
                                hovermode="x unified",
                                showlegend=True
                            )
                            st.plotly_chart(fig_trading, use_container_width=True)

                            # Identificar la MA más "viable" para trading
                            trading_df['Score'] = (
                                (trading_df['Buy_Success_Rate (%)'] * trading_df['Avg_Buy_Gain (%)']) +
                                (trading_df['Sell_Success_Rate (%)'] * abs(trading_df['Avg_Sell_Gain (%)']))
                            ) / 2
                            if trading_df['Score'].notna().any():
                                best_ma = trading_df.loc[trading_df['Score'].idxmax()]
                                st.markdown(f"""
                                ### ¿Cuál es la mejor {ma_type_ma} para esta estrategia?
                                Basado en los datos, la {ma_type_ma} de **{int(best_ma['MA_Length'])} períodos** parece ser la más viable para {ticker_ma} ({compression_ma}). 
                                - **Tasa de Éxito Compra**: {best_ma['Buy_Success_Rate (%)']:.2f}% (el precio sube después de una señal de compra).
                                - **Ganancia Promedio Compra**: {best_ma['Avg_Buy_Gain (%)']:.2f}% en los próximos {look_forward_days} períodos.
                                - **Tasa de Éxito Venta**: {best_ma['Sell_Success_Rate (%)']:.2f}% (el precio baja después de una señal de venta).
                                - **Pérdida Promedio Venta**: {best_ma['Avg_Sell_Gain (%)']:.2f}% en los próximos {look_forward_days} períodos.
                                Esto sugiere que podrías comprar cuando la dispersión cae por debajo del {low_percentile}º percentil y vender cuando sube por encima del {high_percentile}º percentil, con las tasas de éxito y ganancias promedio indicadas.
                                """)
                            else:
                                st.warning("No se pudo determinar una MA óptima porque no hay suficientes señales válidas para calcular un puntaje.")
    else:
        st.warning("⚠️ Por favor, ingrese un símbolo de ticker válido para comenzar el análisis.")

# Footer
st.markdown("---")
st.markdown("© 2024 MTaurus. Todos los derechos reservados.")

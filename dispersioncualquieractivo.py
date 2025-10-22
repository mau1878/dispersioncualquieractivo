import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from functools import lru_cache

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Análisis de Medias Móviles y Dispersión de Precios",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Función compartida para descargar, aplanar y comprimir datos (con caché)
@lru_cache(maxsize=10)
def download_and_compress_data(ticker, start, end, compression='Daily'):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        
        # Aplanar columnas si es MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
        # Renombrar columnas para consistencia
        rename_map = {
            'Open': f'Open_{ticker}',
            'High': f'High_{ticker}',
            'Low': f'Low_{ticker}',
            'Close': f'Close_{ticker}',
            'Adj Close': f'Adj_Close_{ticker}',
            'Volume': f'Volume_{ticker}'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Compresión
        rule_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
        rule = rule_map.get(compression, 'D')
        if rule != 'D':
            agg_dict = {
                col: 'first' if 'Open' in col else
                'max' if 'High' in col else
                'min' if 'Low' in col else
                'last' if 'Close' in col or 'Adj_Close' in col else
                'sum' if 'Volume' in col else 'last'
                for col in df.columns
            }
            df = df.resample(rule).agg(agg_dict).dropna(how='all')
        
        return df
    except Exception as e:
        st.error(f"Error downloading {ticker}: {str(e)}")
        return None

# Función para aplicar ajuste de ratio (opcional)
def apply_ratio_adjustment(data, ticker, start, end, compression):
    ypfd_ba_ticker = "YPFD.BA"
    ypf_ticker = "YPF"
    ypfd_ba_data = download_and_compress_data(ypfd_ba_ticker, start, end, compression)
    ypf_data = download_and_compress_data(ypf_ticker, start, end, compression)
    
    if ypfd_ba_data is None or ypf_data is None:
        raise ValueError("Failed to download ratio data")
    
    close_ypfd = f'Close_{ypfd_ba_ticker}'
    close_ypf = f'Close_{ypf_ticker}'
    
    if close_ypfd not in ypfd_ba_data.columns or close_ypf not in ypf_data.columns:
        raise ValueError("Missing close columns for ratio")
    
    ypfd_ba_data = ypfd_ba_data.fillna(method='ffill').fillna(method='bfill')
    ypf_data = ypf_data.fillna(method='ffill').fillna(method='bfill')
    
    ratio = ypfd_ba_data[close_ypfd] / ypf_data[close_ypf]
    ratio = ratio.reindex(data.index).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    close_col = f'Close_{ticker}'
    data['Close_Adjusted'] = data[close_col] / ratio
    return data

# Función para calcular medias móviles (vectorizada)
def calculate_moving_average(series, ma_type, ma_length):
    if ma_type == "SMA":
        return series.rolling(window=ma_length).mean()
    elif ma_type == "EMA":
        return series.ewm(span=ma_length, adjust=False).mean()
    elif ma_type == "WMA":
        weights = np.arange(1, ma_length + 1)
        def wma_func(x):
            return np.dot(x, weights) / weights.sum() if len(x) == ma_length else np.nan
        return series.rolling(window=ma_length).apply(wma_func, raw=True)
    return pd.Series(np.nan, index=series.index)

# Función mejorada para análisis de estrategia (sin lookahead bias, vectorizado donde posible)
def analyze_ma_percentile_strategy(data, price_column, ma_lengths, ma_type, look_forward_days, low_percentile, high_percentile, min_signals=10):
    results = []
    data = data.copy()  # Evitar mutación
    
    # Precomputar forward returns para eficiencia
    data['forward_max'] = data[price_column].rolling(window=look_forward_days + 1).max().shift(-look_forward_days)
    data['forward_min'] = data[price_column].rolling(window=look_forward_days + 1).min().shift(-look_forward_days)
    data['forward_return_max'] = (data['forward_max'] - data[price_column]) / data[price_column] * 100
    data['forward_return_min'] = (data['forward_min'] - data[price_column]) / data[price_column] * 100
    
    for ma_length in ma_lengths:
        ma_label = f'{ma_type}_{ma_length}'
        data[ma_label] = calculate_moving_average(data[price_column], ma_type, ma_length)
        data['Dispersion'] = (data[price_column] - data[ma_label]) / data[ma_label] * 100
        
        # Calcular percentiles rolling para evitar lookahead bias
        rolling_low = data['Dispersion'].expanding().quantile(low_percentile / 100)
        rolling_high = data['Dispersion'].expanding().quantile(high_percentile / 100)
        
        # Señales vectorizadas
        buy_mask = data['Dispersion'] <= rolling_low
        sell_mask = data['Dispersion'] >= rolling_high
        
        buy_signals = buy_mask.sum()
        sell_signals = sell_mask.sum()
        
        if buy_signals + sell_signals < min_signals:
            continue  # Saltar si no hay suficientes señales
        
        buy_successes = ((data['forward_max'] > data[price_column]) & buy_mask).sum()
        sell_successes = ((data['forward_min'] < data[price_column]) & sell_mask).sum()
        
        buy_success_rate = (buy_successes / buy_signals * 100) if buy_signals > 0 else 0
        sell_success_rate = (sell_successes / sell_signals * 100) if sell_signals > 0 else 0
        
        avg_buy_gain = data.loc[buy_mask, 'forward_return_max'].mean() if buy_signals > 0 else 0
        avg_sell_return = data.loc[sell_mask, 'forward_return_min'].mean() if sell_signals > 0 else 0
        
        # Puntaje mejorado (normalizado por número de señales)
        score = (
            (buy_success_rate * avg_buy_gain * buy_signals) +
            (sell_success_rate * abs(avg_sell_return) * sell_signals)
        ) / (buy_signals + sell_signals + 1e-6)  # Evitar división por cero
        
        results.append({
            'MA_Length': ma_length,
            'Buy_Signals': buy_signals,
            'Buy_Success_Rate (%)': buy_success_rate,
            'Avg_Buy_Gain (%)': avg_buy_gain,
            'Sell_Signals': sell_signals,
            'Sell_Success_Rate (%)': sell_success_rate,
            'Avg_Sell_Return (%)': avg_sell_return,
            'Score': score
        })
    
    df = pd.DataFrame(results)
    if df.empty:
        st.warning("No sufficient signals found for any MA length.")
    return df

# Título de la aplicación
st.title("📈 Análisis de Medias Móviles y Dispersión de Precios - MTaurus")

# Crear pestañas
tab1, tab2 = st.tabs(["Análisis Original", "Análisis de Trading con Percentiles de Dispersión"])

# Función compartida para inputs de fechas y compresión
def get_date_and_compression(key_suffix):
    start_date = st.date_input(
        "📅 Fecha de inicio",
        value=pd.to_datetime('2000-01-01'),
        min_value=pd.to_datetime('1900-01-01'),
        max_value=pd.to_datetime('today'),
        key=f"start_{key_suffix}"
    )
    end_date = st.date_input(
        "📅 Fecha de fin",
        value=pd.to_datetime('today'),
        min_value=pd.to_datetime('1900-01-01'),
        max_value=pd.to_datetime('today'),
        key=f"end_{key_suffix}"
    )
    compression = st.selectbox("📅 Compresión de datos", ["Daily", "Weekly", "Monthly"], key=f"compression_{key_suffix}")
    return pd.to_datetime(start_date), pd.to_datetime(end_date), compression

# Pestaña 1: Análisis Original
with tab1:
    ticker = st.text_input("🖊️ Ticker", value="AAPL").upper()
    if ticker:
        ma_type = st.selectbox("📊 Tipo de MA", ["SMA", "EMA", "WMA"])
        ma_window = st.number_input("📊 Ventana de MA", min_value=2, value=21)  # Min 2 para evitar trivial
        start, end, compression = get_date_and_compression("original")
        apply_ratio = st.checkbox("🔄 Ajustar por ratio YPFD.BA/YPF")
        
        if start > end:
            st.error("Fecha de inicio > fin")
        else:
            with st.spinner("Descargando datos..."):
                data = download_and_compress_data(ticker, start, end, compression)
            
            if data is not None:
                close_col = f'Close_{ticker}'
                if apply_ratio:
                    try:
                        data = apply_ratio_adjustment(data, ticker, start, end, compression)
                        price_column = 'Close_Adjusted'
                    except ValueError as e:
                        st.error(str(e))
                        price_column = close_col
                else:
                    price_column = close_col
                
                if price_column not in data.columns:
                    st.error(f"Columna {price_column} no existe")
                else:
                    ma_label = f'{ma_type}_{ma_window}'
                    data[ma_label] = calculate_moving_average(data[price_column], ma_type, ma_window)
                    data['Porcentaje_Dispersión'] = (data[price_column] - data[ma_label]) / data[ma_label] * 100
                    
                    # Gráfico de precio con MA
                    st.subheader(f"📈 Precio Histórico con {ma_type} ({compression})")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name='Precio'))
                    fig.add_trace(go.Scatter(x=data.index, y=data[ma_label], mode='lines', name=ma_label))
                    fig.update_layout(title=f"Precio de {ticker}", xaxis_title="Fecha", yaxis_title="Precio", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Gráfico de dispersión (simplificado, removí histograma Matplotlib por consistencia con Plotly)
                    st.subheader(f"📉 Dispersión Histórica ({compression})")
                    fig_disp = go.Figure(go.Scatter(x=data.index, y=data['Porcentaje_Dispersión'], mode='lines', name='Dispersión %'))
                    historical_mean = data['Porcentaje_Dispersión'].mean()
                    fig_disp.add_hline(y=historical_mean, line_dash="dash", annotation_text=f"Mean: {historical_mean:.2f}%")
                    fig_disp.add_hline(y=0, line_color="red")
                    lower_percentile = st.slider("Percentil inferior", 1, 49, 5)
                    upper_percentile = st.slider("Percentil superior", 51, 99, 95)
                    disp_data = data['Porcentaje_Dispersión'].dropna()
                    if not disp_data.empty:
                        low_val = np.percentile(disp_data, lower_percentile)
                        high_val = np.percentile(disp_data, upper_percentile)
                        fig_disp.add_hline(y=low_val, line_dash="dash", line_color="red", annotation_text=f"P{lower_percentile}: {low_val:.2f}%")
                        fig_disp.add_hline(y=high_val, line_dash="dash", line_color="green", annotation_text=f"P{upper_percentile}: {high_val:.2f}%")
                    fig_disp.update_layout(title=f"Dispersión de {ticker}", template="plotly_dark")
                    st.plotly_chart(fig_disp, use_container_width=True)
                    
                    # Histograma con Plotly
                    st.subheader("📊 Histograma de Dispersión")
                    num_bins = st.slider("Bins", 10, 100, 50)
                    fig_hist = go.Figure(go.Histogram(x=disp_data, nbinsx=num_bins))
                    percentiles = [5,15,25,50,75,85,95]
                    perc_vals = np.percentile(disp_data, percentiles)
                    for p, v in zip(percentiles, perc_vals):
                        fig_hist.add_vline(x=v, line_dash="dash", annotation_text=f"P{p}")
                    fig_hist.update_layout(title="Histograma", template="plotly_dark")
                    st.plotly_chart(fig_hist)

# Pestaña 2: Análisis de Trading
with tab2:
    st.header("Análisis de Trading con Percentiles de Dispersión")
    st.markdown("""Esta herramienta evalúa MAs para una estrategia basada en dispersión extrema. 
    Usamos percentiles expanding para evitar bias. Métricas usan max/min forward para estimar potencial, pero nota: esto es optimista y no un backtest real.""")
    
    ticker_ma = st.text_input("🖊️ Ticker", value="AAPL").upper()
    if ticker_ma:
        ma_type_ma = st.selectbox("📊 Tipo de MA", ["SMA", "EMA", "WMA"])
        start_ma, end_ma, compression_ma = get_date_and_compression("ma")
        min_ma = st.number_input("Min MA length", min_value=2, value=5)
        max_ma = st.number_input("Max MA length", min_value=min_ma+1, value=50)
        step_ma = st.number_input("Paso MA", min_value=1, value=5)
        look_forward = st.number_input("Períodos forward", min_value=1, value=5)
        low_perc = st.slider("Percentil bajo (compra)", 1, 49, 5)
        high_perc = st.slider("Percentil alto (venta)", 51, 99, 95)
        apply_ratio_ma = st.checkbox("🔄 Ajustar por ratio")
        
        if start_ma > end_ma:
            st.error("Fecha de inicio > fin")
        elif st.button("Analizar"):
            with st.spinner("Procesando..."):
                data_ma = download_and_compress_data(ticker_ma, start_ma, end_ma, compression_ma)
                if data_ma is not None:
                    close_col = f'Close_{ticker_ma}'
                    if apply_ratio_ma:
                        try:
                            data_ma = apply_ratio_adjustment(data_ma, ticker_ma, start_ma, end_ma, compression_ma)
                            price_column_ma = 'Close_Adjusted'
                        except ValueError as e:
                            st.error(str(e))
                            price_column_ma = close_col
                    else:
                        price_column_ma = close_col
                    
                    if price_column_ma in data_ma.columns and len(data_ma) > look_forward + max_ma:
                        ma_lengths = range(min_ma, max_ma + 1, step_ma)
                        trading_df = analyze_ma_percentile_strategy(
                            data_ma, price_column_ma, ma_lengths, ma_type_ma, look_forward, low_perc, high_perc
                        )
                        if not trading_df.empty:
                            st.subheader("Resultados")
                            st.dataframe(trading_df.style.format("{:.2f}"))
                            
                            # Gráfico
                            fig_trade = go.Figure()
                            fig_trade.add_trace(go.Scatter(x=trading_df['MA_Length'], y=trading_df['Buy_Success_Rate (%)'], name='Buy Success %', mode='lines+markers'))
                            fig_trade.add_trace(go.Scatter(x=trading_df['MA_Length'], y=trading_df['Sell_Success_Rate (%)'], name='Sell Success %', mode='lines+markers'))
                            fig_trade.add_trace(go.Scatter(x=trading_df['MA_Length'], y=trading_df['Avg_Buy_Gain (%)'], name='Avg Buy Gain %', mode='lines+markers', line_dash='dash'))
                            fig_trade.add_trace(go.Scatter(x=trading_df['MA_Length'], y=trading_df['Avg_Sell_Return (%)'], name='Avg Sell Return %', mode='lines+markers', line_dash='dash'))
                            fig_trade.update_layout(title="Métricas por Longitud MA", template="plotly_dark")
                            st.plotly_chart(fig_trade)
                            
                            # Mejor MA
                            if 'Score' in trading_df.columns:
                                best = trading_df.loc[trading_df['Score'].idxmax()]
                                st.markdown(f"Mejor MA: {best['MA_Length']} períodos (Score: {best['Score']:.2f})")
                    else:
                        st.error("Datos insuficientes")

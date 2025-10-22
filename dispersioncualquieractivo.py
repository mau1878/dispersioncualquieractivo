import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Análisis de Medias Móviles", layout="wide", initial_sidebar_state="expanded")

# Función para descargar y comprimir datos
@st.cache_data
def download_and_compress_data(ticker, start, end, compression='Daily'):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        
        # Aplanar columnas MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
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
            agg_dict = {col: 'first' if 'Open' in col else 'max' if 'High' in col else 'min' if 'Low' in col else
                       'last' if 'Close' in col or 'Adj_Close' in col else 'sum' if 'Volume' in col else 'last'
                       for col in df.columns}
            df = df.resample(rule).agg(agg_dict).dropna(how='all')
        
        return df
    except Exception as e:
        st.error(f"Error downloading {ticker}: {str(e)}")
        return None

# Función para aplicar ajuste de ratio
def apply_ratio_adjustment(data, ticker, start, end, compression):
    ypfd_ba_ticker, ypf_ticker = "YPFD.BA", "YPF"
    ypfd_ba_data = download_and_compress_data(ypfd_ba_ticker, start, end, compression)
    ypf_data = download_and_compress_data(ypf_ticker, start, end, compression)
    
    if ypfd_ba_data is None or ypf_data is None:
        raise ValueError("Failed to download ratio data")
    
    close_ypfd, close_ypf = f'Close_{ypfd_ba_ticker}', f'Close_{ypf_ticker}'
    if close_ypfd not in ypfd_ba_data.columns or close_ypf not in ypf_data.columns:
        raise ValueError("Missing close columns for ratio")
    
    ypfd_ba_data = ypfd_ba_data.fillna(method='ffill').fillna(method='bfill')
    ypf_data = ypf_data.fillna(method='ffill').fillna(method='bfill')
    
    ratio = ypfd_ba_data[close_ypfd] / ypf_data[close_ypf]
    ratio = ratio.reindex(data.index).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    data['Close_Adjusted'] = data[f'Close_{ticker}'] / ratio
    return data

# Función para calcular medias móviles
def calculate_moving_average(series, ma_type, ma_length):
    if ma_type == "SMA":
        return series.rolling(window=ma_length).mean()
    elif ma_type == "EMA":
        return series.ewm(span=ma_length, adjust=False).mean()
    elif ma_type == "WMA":
        weights = np.arange(1, ma_length + 1)
        return series.rolling(window=ma_length).apply(lambda x: np.dot(x, weights) / weights.sum() if len(x) == ma_length else np.nan, raw=True)
    return pd.Series(np.nan, index=series.index)

# Función para análisis de estrategia (sin lookahead bias)
def analyze_ma_percentile_strategy(data, price_column, ma_lengths, ma_type, look_forward_days, low_percentile, high_percentile, min_signals=10):
    results = []
    data = data.copy()
    
    # Precomputar retornos forward
    data['forward_price'] = data[price_column].shift(-look_forward_days)
    data['forward_return'] = (data['forward_price'] - data[price_column]) / data[price_column] * 100
    
    for ma_length in ma_lengths:
        ma_label = f'{ma_type}_{ma_length}'
        data[ma_label] = calculate_moving_average(data[price_column], ma_type, ma_length)
        data['Dispersion'] = (data[price_column] - data[ma_label]) / data[ma_label] * 100
        
        # Percentiles sin lookahead
        rolling_low = data['Dispersion'].expanding(min_periods=ma_length*2).quantile(low_percentile / 100)
        rolling_high = data['Dispersion'].expanding(min_periods=ma_length*2).quantile(high_percentile / 100)
        
        buy_mask = data['Dispersion'] <= rolling_low
        sell_mask = data['Dispersion'] >= rolling_high
        
        buy_signals = buy_mask.sum()
        sell_signals = sell_mask.sum()
        
        if buy_signals + sell_signals < min_signals:
            continue
        
        buy_successes = ((data['forward_price'] > data[price_column]) & buy_mask).sum()
        sell_successes = ((data['forward_price'] < data[price_column]) & sell_mask).sum()
        
        buy_success_rate = (buy_successes / buy_signals * 100) if buy_signals > 0 else 0
        sell_success_rate = (sell_successes / sell_signals * 100) if sell_signals > 0 else 0
        
        avg_buy_gain = data.loc[buy_mask, 'forward_return'].mean() if buy_signals > 0 else 0
        avg_sell_return = data.loc[sell_mask, 'forward_return'].mean() if sell_signals > 0 else 0
        
        score = (buy_success_rate * avg_buy_gain * buy_signals + sell_success_rate * abs(avg_sell_return) * sell_signals) / (buy_signals + sell_signals + 1e-6)
        
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
        st.warning("No sufficient signals found.")
    return df

# Inputs compartidos
def get_common_inputs(key_suffix):
    with st.expander("Configuración de Datos"):
        start = st.date_input("Fecha de inicio", value=pd.to_datetime('2020-01-01'), min_value=pd.to_datetime('1900-01-01'), max_value=pd.to_datetime('today'), key=f"start_{key_suffix}", help="Fecha inicial para datos históricos")
        end = st.date_input("Fecha de fin", value=pd.to_datetime('today'), min_value=pd.to_datetime('1900-01-01'), max_value=pd.to_datetime('today'), key=f"end_{key_suffix}", help="Fecha final para datos")
        compression = st.selectbox("Compresión", ["Daily", "Weekly", "Monthly"], key=f"compression_{key_suffix}", help="Daily: cada día; Weekly: cada semana; Monthly: cada mes")
        apply_ratio = st.checkbox("Ajustar por ratio YPFD.BA/YPF", key=f"ratio_{key_suffix}", help="Ajusta precios dividiendo por el ratio de YPFD.BA/YPF (mercado argentino)")
    return pd.to_datetime(start), pd.to_datetime(end), compression, apply_ratio

# Título
st.title("📈 Análisis de Medias Móviles y Dispersión")

# Pestañas
tab1, tab2 = st.tabs(["Análisis de Precios", "Análisis de Trading"])

# Tab 1: Análisis de Precios
with tab1:
    ticker = st.text_input("Ticker", value="AAPL", key="ticker_original").upper()
    if ticker:
        with st.expander("Configuración de Media Móvil"):
            ma_type = st.selectbox("Tipo de MA", ["SMA", "EMA", "WMA"], key="ma_type_original", help="SMA: promedio simple; EMA: exponencial; WMA: ponderado")
            ma_window = st.number_input("Ventana MA", min_value=2, value=21, key="ma_window_original", help="Períodos para calcular la media móvil")
        
        start, end, compression, apply_ratio = get_common_inputs("original")
        
        if start > end:
            st.error("Fecha de inicio posterior a fin")
        else:
            with st.spinner("Cargando datos..."):
                data = download_and_compress_data(ticker, start, end, compression)
            
            if data is not None:
                close_col = f'Close_{ticker}'
                price_column = close_col
                if apply_ratio:
                    try:
                        data = apply_ratio_adjustment(data, ticker, start, end, compression)
                        price_column = 'Close_Adjusted'
                    except ValueError as e:
                        st.error(str(e))
                
                if price_column in data.columns:
                    ma_label = f'{ma_type}_{ma_window}'
                    data[ma_label] = calculate_moving_average(data[price_column], ma_type, ma_window)
                    data['Dispersion_%'] = (data[price_column] - data[ma_label]) / data[ma_label] * 100
                    
                    # Gráfico de precios
                    st.subheader(f"Precio con {ma_type}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name='Precio', line_color='#1f77b4'))
                    fig.add_trace(go.Scatter(x=data.index, y=data[ma_label], mode='lines', name=ma_label, line_color='#ff7f0e'))
                    fig.update_layout(title=f"{ticker} Precio ({compression})", xaxis_title="Fecha", yaxis_title="Precio", template="plotly_dark", showlegend=True, xaxis_gridcolor='gray', yaxis_gridcolor='gray')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Gráfico de dispersión
                    st.subheader(f"Dispersión Histórica")
                    fig_disp = go.Figure(go.Scatter(x=data.index, y=data['Dispersion_%'], mode='lines', name='Dispersión %', line_color='#d62728'))
                    mean_disp = data['Dispersion_%'].mean()
                    fig_disp.add_hline(y=mean_disp, line_dash="dash", line_color="#17becf", annotation_text=f"Media: {mean_disp:.2f}%")
                    fig_disp.add_hline(y=0, line_color="white")
                    disp_data = data['Dispersion_%'].dropna()
                    if not disp_data.empty:
                        low_val = np.percentile(disp_data, 5)
                        high_val = np.percentile(disp_data, 95)
                        fig_disp.add_hline(y=low_val, line_dash="dash", line_color="#ff9896", annotation_text=f"P5: {low_val:.2f}%")
                        fig_disp.add_hline(y=high_val, line_dash="dash", line_color="#98df8a", annotation_text=f"P95: {high_val:.2f}%")
                    fig_disp.update_layout(title=f"Dispersión de {ticker}", xaxis_title="Fecha", yaxis_title="Dispersión (%)", template="plotly_dark", showlegend=True)
                    st.plotly_chart(fig_disp, use_container_width=True)

# Tab 2: Análisis de Trading
with tab2:
    st.markdown("Evalúa señales de compra/venta basadas en dispersión extrema respecto a medias móviles, usando retornos a plazo fijo (no máximo/mínimo).")
    ticker_ma = st.text_input("Ticker", value="AAPL", key="ticker_ma").upper()
    if ticker_ma:
        with st.expander("Configuración de Estrategia"):
            ma_type_ma = st.selectbox("Tipo de MA", ["SMA", "EMA", "WMA"], key="ma_type_ma")
            min_ma = st.number_input("Mínima longitud MA", min_value=2, value=5, key="min_ma")
            max_ma = st.number_input("Máxima longitud MA", min_value=min_ma+1, value=50, key="max_ma")
            step_ma = st.number_input("Paso MA", min_value=1, value=5, key="step_ma")
            look_forward = st.number_input(f"Períodos forward ({'días' if compression == 'Daily' else 'semanas' if compression == 'Weekly' else 'meses'})", min_value=1, value=5, key="look_forward", help="Períodos para evaluar retornos")
            low_perc = st.slider("Percentil bajo (compra)", 1, 49, 5, key="low_perc_ma")
            high_perc = st.slider("Percentil alto (venta)", 51, 99, 95, key="high_perc_ma")
        
        start_ma, end_ma, compression_ma, apply_ratio_ma = get_common_inputs("ma")
        
        if start_ma > end_ma:
            st.error("Fecha de inicio posterior a fin")
        elif st.button("Analizar", key="analyze_ma"):
            with st.spinner("Analizando..."):
                data_ma = download_and_compress_data(ticker_ma, start_ma, end_ma, compression_ma)
                if data_ma is not None and len(data_ma) > look_forward + max_ma:
                    price_column_ma = f'Close_{ticker_ma}'
                    if apply_ratio_ma:
                        try:
                            data_ma = apply_ratio_adjustment(data_ma, ticker_ma, start_ma, end_ma, compression_ma)
                            price_column_ma = 'Close_Adjusted'
                        except ValueError as e:
                            st.error(str(e))
                    
                    if price_column_ma in data_ma.columns:
                        ma_lengths = range(min_ma, max_ma + 1, step_ma)
                        trading_df = analyze_ma_percentile_strategy(
                            data_ma, price_column_ma, ma_lengths, ma_type_ma, look_forward, low_perc, high_perc
                        )
                        if not trading_df.empty:
                            st.subheader("Resultados")
                            st.dataframe(trading_df.style.format("{:.2f}"), use_container_width=True)
                            
                            # Gráficos en subplots
                            fig_trade = go.Figure()
                            fig_trade.add_trace(go.Scatter(x=trading_df['MA_Length'], y=trading_df['Buy_Success_Rate (%)'], name='Buy Success %', mode='lines+markers', line_color='#98df8a'))
                            fig_trade.add_trace(go.Scatter(x=trading_df['MA_Length'], y=trading_df['Sell_Success_Rate (%)'], name='Sell Success %', mode='lines+markers', line_color='#ff9896'))
                            fig_trade.update_layout(title="Tasa de Éxito por Longitud MA", xaxis_title="Períodos MA", yaxis_title="Tasa de Éxito (%)", template="plotly_dark", showlegend=True)
                            st.plotly_chart(fig_trade, use_container_width=True)
                            
                            fig_gains = go.Figure()
                            fig_gains.add_trace(go.Scatter(x=trading_df['MA_Length'], y=trading_df['Avg_Buy_Gain (%)'], name='Avg Buy Gain %', mode='lines+markers', line_color='#1f77b4', line_dash='dash'))
                            fig_gains.add_trace(go.Scatter(x=trading_df['MA_Length'], y=trading_df['Avg_Sell_Return (%)'], name='Avg Sell Return %', mode='lines+markers', line_color='#ff7f0e', line_dash='dash'))
                            fig_gains.update_layout(title="Retornos Promedio por Longitud MA", xaxis_title="Períodos MA", yaxis_title="Retorno (%)", template="plotly_dark", showlegend=True)
                            st.plotly_chart(fig_gains, use_container_width=True)
                            
                            if 'Score' in trading_df.columns and trading_df['Buy_Signals'].sum() + trading_df['Sell_Signals'].sum() > 50:
                                best = trading_df.loc[trading_df['Score'].idxmax()]
                                st.markdown(f"**Mejor MA**: {int(best['MA_Length'])} períodos\n- Buy Success: {best['Buy_Success_Rate (%)']:.2f}%\n- Avg Buy Gain: {best['Avg_Buy_Gain (%)']:.2f}%\n- Sell Success: {best['Sell_Success_Rate (%)']:.2f}%\n- Avg Sell Return: {best['Avg_Sell_Return (%)']:.2f}%")
                                st.warning("Resultados históricos no garantizan rendimiento futuro.")

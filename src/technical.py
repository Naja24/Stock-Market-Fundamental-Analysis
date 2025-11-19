# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import plotly.graph_objects as go
# from sklearn.linear_model import LinearRegression
# import numpy as np
# from ta.trend import SMAIndicator, EMAIndicator
# from ta.momentum import RSIIndicator

# def display_technical_analysis(ticker_symbol):
#     st.header(f"Technical Analysis & Prediction: {ticker_symbol}")
    
#     # Fetch Data
#     data = yf.download(ticker_symbol, period="5y", interval="1d")
#     if data.empty:
#         st.error("No data found. Please check the ticker.")
#         return

#     # --- CHARTING ---
#     st.subheader("Price Chart with Indicators")
    
#     # Calculate Indicators
#     # Ensure we pass 1-dimensional arrays/Series to TA functions
#     close_1d = data['Close'].squeeze()

#     data['SMA_50'] = SMAIndicator(close_1d, window=50).sma_indicator()
#     data['EMA_20'] = EMAIndicator(close_1d, window=20).ema_indicator()
#     data['RSI'] = RSIIndicator(close_1d).rsi()
    
#     # Plotly Chart
#     # Ensure plotting arrays are 1D (some upstream operations may create (n,1) arrays)
#     open_1d = data['Open'].squeeze()
#     high_1d = data['High'].squeeze()
#     low_1d = data['Low'].squeeze()
#     close_1d = data['Close'].squeeze()

#     fig = go.Figure()
#     fig.add_trace(go.Candlestick(x=data.index,
#                 open=open_1d, high=high_1d,
#                 low=low_1d, close=close_1d, name='OHLC'))
#     fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'].squeeze(), mode='lines', name='SMA 50', line=dict(color='orange')))
#     fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'].squeeze(), mode='lines', name='EMA 20', line=dict(color='blue')))
    
#     fig.update_layout(title=f'{ticker_symbol} Price History', xaxis_title='Date', yaxis_title='Price', height=600)
#     st.plotly_chart(fig, use_container_width=True)

#     # RSI Metric
#     last_rsi = data['RSI'].iloc[-1]
#     st.metric("Current RSI (14)", f"{last_rsi:.2f}", delta=None)
#     if last_rsi > 70:
#         st.warning("Stock is Overbought (RSI > 70)")
#     elif last_rsi < 30:
#         st.success("Stock is Oversold (RSI < 30)")
#     else:
#         st.info("Stock is in Neutral Zone")

#     # --- PREDICTION MODEL ---
#     st.subheader("🔮 Future Price Prediction (AI Model)")
#     st.caption("Using a lightweight Linear Regression model for demonstration. (Reference to Transformer models in GitHub repo)")

#     # Data Prep for Prediction
#     df_pred = data[['Close']].dropna().reset_index()
#     df_pred['Date_Ordinal'] = df_pred['Date'].map(pd.Timestamp.toordinal)
    
#     X = df_pred[['Date_Ordinal']]
#     y = df_pred['Close']
    
#     # Train Model
#     model = LinearRegression()
#     model.fit(X, y)
    
#     # Predict Next 7 Days
#     future_days = 7
#     last_date = df_pred['Date'].iloc[-1]
#     future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_days+1)]
#     future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    
#     predictions = model.predict(future_ordinals)
#     # Ensure predictions is 1D for DataFrame construction
#     predictions = np.ravel(predictions)
    
#     # Display Predictions
#     pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.write("### Forecasted Prices (Next 7 Days)")
#         st.dataframe(pred_df.style.format({"Predicted Price": "₹ {:.2f}"}))
    
#     with col2:
#         current_price = y.iloc[-1]
#         predicted_price = predictions[-1]
#         change = ((predicted_price - current_price) / current_price) * 100
        
#         st.metric("7-Day Price Target", f"₹ {predicted_price:.2f}", f"{change:.2f}%")
        
#         if change > 0:
#             st.success("Trend Recommendation: BUY / HOLD")
#         else:
#             st.error("Trend Recommendation: SELL / CAUTION")

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def get_trading_days(start_date, num_days=7):
    """Generate future trading days (excluding weekends)"""
    trading_days = []
    current_date = start_date
    while len(trading_days) < num_days:
        current_date += timedelta(days=1)
        # Skip weekends (Saturday=5, Sunday=6)
        if current_date.weekday() < 5:
            trading_days.append(current_date)
    return trading_days

def display_technical_analysis(ticker_symbol):
    st.header(f"📈 Technical Analysis & Prediction: {ticker_symbol}")
    
    # Fetch Data with progress indicator
    with st.spinner('Fetching market data...'):
        data = yf.download(ticker_symbol, period="2y", interval="1d", progress=False)
    
    if data.empty:
        st.error("❌ No data found. Please check the ticker symbol.")
        return

    # Remove any multi-level column indexes
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Ensure we have enough data
    if len(data) < 50:
        st.warning("⚠️ Limited data available. Results may be less accurate.")
    
    # --- CHARTING ---
    st.subheader("📊 Price Chart with Technical Indicators")
    
    # Calculate Indicators
    close_series = data['Close'].squeeze()
    
    # Ensure sufficient data for indicators
    if len(close_series) >= 50:
        data['SMA_50'] = SMAIndicator(close_series, window=50).sma_indicator()
    if len(close_series) >= 20:
        data['EMA_20'] = EMAIndicator(close_series, window=20).ema_indicator()
    if len(close_series) >= 14:
        data['RSI'] = RSIIndicator(close_series, window=14).rsi()
    
    # Create Plotly Chart
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'].squeeze(),
        high=data['High'].squeeze(),
        low=data['Low'].squeeze(),
        close=data['Close'].squeeze(),
        name='OHLC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # Add moving averages if available
    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_50'].squeeze(),
            mode='lines',
            name='SMA 50',
            line=dict(color='#ff9800', width=2)
        ))
    
    if 'EMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['EMA_20'].squeeze(),
            mode='lines',
            name='EMA 20',
            line=dict(color='#2196f3', width=2)
        ))
    
    fig.update_layout(
        title=f'{ticker_symbol} Price History',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        height=600,
        template='plotly_white',
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- RSI ANALYSIS ---
    st.subheader("📉 Relative Strength Index (RSI)")
    
    if 'RSI' in data.columns and not data['RSI'].isna().all():
        col1, col2, col3 = st.columns(3)
        
        last_rsi = data['RSI'].iloc[-1]
        prev_rsi = data['RSI'].iloc[-2] if len(data) > 1 else last_rsi
        rsi_change = last_rsi - prev_rsi
        
        with col1:
            st.metric("Current RSI (14)", f"{last_rsi:.2f}", f"{rsi_change:+.2f}")
        
        with col2:
            if last_rsi > 70:
                st.error("🔴 Overbought (RSI > 70)")
                signal = "Consider Selling"
            elif last_rsi < 30:
                st.success("🟢 Oversold (RSI < 30)")
                signal = "Consider Buying"
            else:
                st.info("⚪ Neutral Zone")
                signal = "Hold Position"
            st.write(f"**Signal:** {signal}")
        
        with col3:
            # RSI trend
            rsi_7d_avg = data['RSI'].tail(7).mean()
            st.metric("7-Day Avg RSI", f"{rsi_7d_avg:.2f}")
        
        # RSI Chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=data.index[-60:],
            y=data['RSI'].tail(60),
            mode='lines',
            name='RSI',
            line=dict(color='#9c27b0', width=2)
        ))
        
        # Add overbought/oversold lines
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral")
        
        fig_rsi.update_layout(
            title='RSI Trend (Last 60 Days)',
            xaxis_title='Date',
            yaxis_title='RSI Value',
            height=300,
            template='plotly_white',
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
    else:
        st.warning("⚠️ Insufficient data to calculate RSI")

    # --- PREDICTION MODEL ---
    st.subheader("🔮 AI-Powered Price Prediction")
    st.caption("📌 Using Linear Regression for trend forecasting (For demonstration purposes)")

    # Data Prep for Prediction
    df_pred = data[['Close']].dropna().copy()
    
    if len(df_pred) < 30:
        st.error("❌ Insufficient data for prediction")
        return
    
    df_pred = df_pred.reset_index()
    df_pred['Date_Ordinal'] = df_pred['Date'].map(pd.Timestamp.toordinal)
    
    X = df_pred[['Date_Ordinal']].values
    y = df_pred['Close'].values
    
    # Train Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate model confidence (R² score)
    r2_score = model.score(X, y)
    
    # Predict Next 7 Trading Days (excluding weekends)
    last_date = df_pred['Date'].iloc[-1]
    future_dates = get_trading_days(last_date, num_days=7)
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    
    predictions = model.predict(future_ordinals).ravel()
    
    # Display Predictions
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Day': [d.strftime('%A') for d in future_dates],
        'Predicted Price': predictions
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("### 📅 7-Day Price Forecast (Trading Days Only)")
        st.dataframe(
            pred_df.style.format({"Predicted Price": "₹ {:.2f}"})
            .background_gradient(subset=['Predicted Price'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    with col2:
        current_price = y[-1]
        predicted_price = predictions[-1]
        change = ((predicted_price - current_price) / current_price) * 100
        
        st.metric(
            "7-Day Target Price",
            f"₹ {predicted_price:.2f}",
            f"{change:+.2f}%",
            delta_color="normal"
        )
        
        st.metric("Model Confidence", f"{r2_score*100:.1f}%")
        
        if change > 2:
            st.success("📈 **Bullish Trend**\n\nConsider: BUY/HOLD")
        elif change < -2:
            st.error("📉 **Bearish Trend**\n\nConsider: SELL/CAUTION")
        else:
            st.info("➡️ **Sideways Trend**\n\nConsider: HOLD")
    
    # Prediction Chart
    st.subheader("📈 Prediction Visualization")
    
    fig_pred = go.Figure()
    
    # Historical prices (last 30 days)
    historical_last_30 = df_pred.tail(30)
    fig_pred.add_trace(go.Scatter(
        x=historical_last_30['Date'],
        y=historical_last_30['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1976d2', width=3)
    ))
    
    # Predicted prices
    fig_pred.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='#f57c00', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    fig_pred.update_layout(
        title='Price Prediction for Next 7 Trading Days',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Disclaimer
    st.info("⚠️ **Disclaimer:** This prediction is for educational purposes only. Past performance does not guarantee future results. Always do your own research before investing.")
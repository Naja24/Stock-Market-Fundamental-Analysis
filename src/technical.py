# # import streamlit as st
# # import yfinance as yf
# # import pandas as pd
# # import plotly.graph_objects as go
# # from sklearn.linear_model import LinearRegression
# # import numpy as np
# # from ta.trend import SMAIndicator, EMAIndicator
# # from ta.momentum import RSIIndicator

# # def display_technical_analysis(ticker_symbol):
# #     st.header(f"Technical Analysis & Prediction: {ticker_symbol}")
    
# #     # Fetch Data
# #     data = yf.download(ticker_symbol, period="5y", interval="1d")
# #     if data.empty:
# #         st.error("No data found. Please check the ticker.")
# #         return

# #     # --- CHARTING ---
# #     st.subheader("Price Chart with Indicators")
    
# #     # Calculate Indicators
# #     # Ensure we pass 1-dimensional arrays/Series to TA functions
# #     close_1d = data['Close'].squeeze()

# #     data['SMA_50'] = SMAIndicator(close_1d, window=50).sma_indicator()
# #     data['EMA_20'] = EMAIndicator(close_1d, window=20).ema_indicator()
# #     data['RSI'] = RSIIndicator(close_1d).rsi()
    
# #     # Plotly Chart
# #     # Ensure plotting arrays are 1D (some upstream operations may create (n,1) arrays)
# #     open_1d = data['Open'].squeeze()
# #     high_1d = data['High'].squeeze()
# #     low_1d = data['Low'].squeeze()
# #     close_1d = data['Close'].squeeze()

# #     fig = go.Figure()
# #     fig.add_trace(go.Candlestick(x=data.index,
# #                 open=open_1d, high=high_1d,
# #                 low=low_1d, close=close_1d, name='OHLC'))
# #     fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'].squeeze(), mode='lines', name='SMA 50', line=dict(color='orange')))
# #     fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'].squeeze(), mode='lines', name='EMA 20', line=dict(color='blue')))
    
# #     fig.update_layout(title=f'{ticker_symbol} Price History', xaxis_title='Date', yaxis_title='Price', height=600)
# #     st.plotly_chart(fig, use_container_width=True)

# #     # RSI Metric
# #     last_rsi = data['RSI'].iloc[-1]
# #     st.metric("Current RSI (14)", f"{last_rsi:.2f}", delta=None)
# #     if last_rsi > 70:
# #         st.warning("Stock is Overbought (RSI > 70)")
# #     elif last_rsi < 30:
# #         st.success("Stock is Oversold (RSI < 30)")
# #     else:
# #         st.info("Stock is in Neutral Zone")

# #     # --- PREDICTION MODEL ---
# #     st.subheader("🔮 Future Price Prediction (AI Model)")
# #     st.caption("Using a lightweight Linear Regression model for demonstration. (Reference to Transformer models in GitHub repo)")

# #     # Data Prep for Prediction
# #     df_pred = data[['Close']].dropna().reset_index()
# #     df_pred['Date_Ordinal'] = df_pred['Date'].map(pd.Timestamp.toordinal)
    
# #     X = df_pred[['Date_Ordinal']]
# #     y = df_pred['Close']
    
# #     # Train Model
# #     model = LinearRegression()
# #     model.fit(X, y)
    
# #     # Predict Next 7 Days
# #     future_days = 7
# #     last_date = df_pred['Date'].iloc[-1]
# #     future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_days+1)]
# #     future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    
# #     predictions = model.predict(future_ordinals)
# #     # Ensure predictions is 1D for DataFrame construction
# #     predictions = np.ravel(predictions)
    
# #     # Display Predictions
# #     pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})
    
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         st.write("### Forecasted Prices (Next 7 Days)")
# #         st.dataframe(pred_df.style.format({"Predicted Price": "₹ {:.2f}"}))
    
# #     with col2:
# #         current_price = y.iloc[-1]
# #         predicted_price = predictions[-1]
# #         change = ((predicted_price - current_price) / current_price) * 100
        
# #         st.metric("7-Day Price Target", f"₹ {predicted_price:.2f}", f"{change:.2f}%")
        
# #         if change > 0:
# #             st.success("Trend Recommendation: BUY / HOLD")
# #         else:
# #             st.error("Trend Recommendation: SELL / CAUTION")

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

def format_price(val):
    """Format price with rupee symbol"""
    return f"₹ {val:.2f}"

def get_color_for_price(val, min_val, max_val):
    """Get color based on value position in range"""
    if max_val == min_val:
        return '#FFFFFF'
    normalized = (val - min_val) / (max_val - min_val)
    # Red to Yellow to Green gradient
    if normalized < 0.5:
        # Red to Yellow
        r = 255
        g = int(255 * (normalized * 2))
        b = 0
    else:
        # Yellow to Green
        r = int(255 * (1 - (normalized - 0.5) * 2))
        g = 255
        b = 0
    return f'#{r:02x}{g:02x}{b:02x}'

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
        
        # Create styled HTML table instead of using background_gradient
        min_price = predictions.min()
        max_price = predictions.max()
        
        html_table = "<table style='width:100%; border-collapse: collapse;'>"
        html_table += "<thead><tr style='background-color: #f0f0f0;'>"
        html_table += "<th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Date</th>"
        html_table += "<th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Day</th>"
        html_table += "<th style='padding: 10px; text-align: right; border: 1px solid #ddd;'>Predicted Price</th>"
        html_table += "</tr></thead><tbody>"
        
        for idx, row in pred_df.iterrows():
            bg_color = get_color_for_price(row['Predicted Price'], min_price, max_price)
            html_table += f"<tr style='background-color: {bg_color};'>"
            html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Date'].strftime('%Y-%m-%d')}</td>"
            html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Day']}</td>"
            html_table += f"<td style='padding: 10px; text-align: right; border: 1px solid #ddd; font-weight: bold;'>₹ {row['Predicted Price']:.2f}</td>"
            html_table += "</tr>"
        
        html_table += "</tbody></table>"
        
        st.markdown(html_table, unsafe_allow_html=True)
    
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



# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import numpy as np
# from ta.trend import SMAIndicator, EMAIndicator
# from ta.momentum import RSIIndicator
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

# # Deep Learning imports — import defensively to avoid crashing the app when TF can't load
# try:
#     import tensorflow as tf
#     from tensorflow import keras
#     from tensorflow.keras.models import Sequential, Model
#     from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
#     from tensorflow.keras.callbacks import EarlyStopping
#     TF_AVAILABLE = True
#     TF_IMPORT_ERROR = None
# except Exception as _tf_err:
#     # TensorFlow failed to load (missing DLLs, incompatible environment, etc.)
#     tf = None
#     keras = None
#     Sequential = Model = LSTM = GRU = Bidirectional = Dense = Dropout = Input = LayerNormalization = MultiHeadAttention = GlobalAveragePooling1D = EarlyStopping = None
#     TF_AVAILABLE = False
#     TF_IMPORT_ERROR = str(_tf_err)

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# def get_trading_days(start_date, num_days=7):
#     """Generate future trading days (excluding weekends)"""
#     trading_days = []
#     current_date = start_date
#     while len(trading_days) < num_days:
#         current_date += timedelta(days=1)
#         if current_date.weekday() < 5:
#             trading_days.append(current_date)
#     return trading_days

# def prepare_sequences(data, n_steps=60):
#     """Prepare sequences for LSTM/GRU models"""
#     X, y = [], []
#     for i in range(n_steps, len(data)):
#         X.append(data[i-n_steps:i])
#         y.append(data[i])
#     return np.array(X), np.array(y)

# def build_lstm_model(n_steps, n_features):
#     """Build LSTM model"""
#     model = Sequential([
#         LSTM(128, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)),
#         Dropout(0.2),
#         LSTM(64, activation='tanh', return_sequences=True),
#         Dropout(0.2),
#         LSTM(32, activation='tanh'),
#         Dropout(0.2),
#         Dense(16, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

# def build_bidirectional_lstm_model(n_steps, n_features):
#     """Build Bidirectional LSTM model"""
#     model = Sequential([
#         Bidirectional(LSTM(128, activation='tanh', return_sequences=True), input_shape=(n_steps, n_features)),
#         Dropout(0.2),
#         Bidirectional(LSTM(64, activation='tanh', return_sequences=True)),
#         Dropout(0.2),
#         Bidirectional(LSTM(32, activation='tanh')),
#         Dropout(0.2),
#         Dense(16, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

# def build_gru_model(n_steps, n_features):
#     """Build GRU model"""
#     model = Sequential([
#         GRU(128, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)),
#         Dropout(0.2),
#         GRU(64, activation='tanh', return_sequences=True),
#         Dropout(0.2),
#         GRU(32, activation='tanh'),
#         Dropout(0.2),
#         Dense(16, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

# def build_tft_model(n_steps, n_features):
#     """Build Temporal Fusion Transformer-inspired model"""
#     inputs = Input(shape=(n_steps, n_features))
    
#     # Multi-head attention layer
#     attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
#     attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    
#     # LSTM processing
#     lstm_output = LSTM(64, return_sequences=True)(attention_output)
#     lstm_output = Dropout(0.2)(lstm_output)
#     lstm_output = LayerNormalization(epsilon=1e-6)(lstm_output)
    
#     # Global pooling
#     pooled = GlobalAveragePooling1D()(lstm_output)
    
#     # Dense layers
#     dense1 = Dense(32, activation='relu')(pooled)
#     dense1 = Dropout(0.2)(dense1)
#     outputs = Dense(1)(dense1)
    
#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

# def build_tsmixer_model(n_steps, n_features):
#     """Build TSMixer-inspired model"""
#     inputs = Input(shape=(n_steps, n_features))
    
#     # Time mixing
#     x = Dense(128, activation='relu')(inputs)
#     x = LayerNormalization(epsilon=1e-6)(x)
#     x = Dropout(0.2)(x)
    
#     # Feature mixing
#     x = tf.transpose(x, perm=[0, 2, 1])
#     x = Dense(64, activation='relu')(x)
#     x = LayerNormalization(epsilon=1e-6)(x)
#     x = Dropout(0.2)(x)
#     x = tf.transpose(x, perm=[0, 2, 1])
    
#     # Additional mixing layers
#     x = Dense(32, activation='relu')(x)
#     x = LayerNormalization(epsilon=1e-6)(x)
    
#     # Global pooling
#     x = GlobalAveragePooling1D()(x)
    
#     # Output
#     outputs = Dense(1)(x)
    
#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

# def train_and_predict(model, X_train, y_train, X_test, scaler, model_name, epochs=50):
#     """Train model and make predictions"""
#     early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
#     with st.spinner(f'🧠 Training {model_name} model...'):
#         history = model.fit(
#             X_train, y_train,
#             epochs=epochs,
#             batch_size=32,
#             verbose=0,
#             callbacks=[early_stop]
#         )
    
#     # Make predictions
#     predictions = model.predict(X_test, verbose=0)
#     predictions = scaler.inverse_transform(predictions)
    
#     return predictions.flatten(), history

# def calculate_metrics(y_true, y_pred):
#     """Calculate performance metrics"""
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     r2 = r2_score(y_true, y_pred)
#     mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
#     return {
#         'MAE': mae,
#         'RMSE': rmse,
#         'R²': r2,
#         'MAPE': mape
#     }

# def get_color_for_price(val, min_val, max_val):
#     """Get color based on value position in range"""
#     if max_val == min_val:
#         return '#FFFFFF'
#     normalized = (val - min_val) / (max_val - min_val)
#     if normalized < 0.5:
#         r = 255
#         g = int(255 * (normalized * 2))
#         b = 0
#     else:
#         r = int(255 * (1 - (normalized - 0.5) * 2))
#         g = 255
#         b = 0
#     return f'#{r:02x}{g:02x}{b:02x}'

# def display_technical_analysis(ticker_symbol):
#     st.header(f"📈 Advanced AI Technical Analysis & Prediction: {ticker_symbol}")
    
#     # Fetch Data
#     with st.spinner('Fetching market data...'):
#         data = yf.download(ticker_symbol, period="2y", interval="1d", progress=False)
    
#     if data.empty:
#         st.error("❌ No data found. Please check the ticker symbol.")
#         return

#     if isinstance(data.columns, pd.MultiIndex):
#         data.columns = data.columns.get_level_values(0)
    
#     if len(data) < 100:
#         st.warning("⚠️ Limited data available. Results may be less accurate.")
#         return
    
#     # --- CHARTING ---
#     st.subheader("📊 Price Chart with Technical Indicators")
    
#     close_series = data['Close'].squeeze()
    
#     if len(close_series) >= 50:
#         data['SMA_50'] = SMAIndicator(close_series, window=50).sma_indicator()
#     if len(close_series) >= 20:
#         data['EMA_20'] = EMAIndicator(close_series, window=20).ema_indicator()
#     if len(close_series) >= 14:
#         data['RSI'] = RSIIndicator(close_series, window=14).rsi()
    
#     fig = go.Figure()
    
#     fig.add_trace(go.Candlestick(
#         x=data.index,
#         open=data['Open'].squeeze(),
#         high=data['High'].squeeze(),
#         low=data['Low'].squeeze(),
#         close=data['Close'].squeeze(),
#         name='OHLC',
#         increasing_line_color='#26a69a',
#         decreasing_line_color='#ef5350'
#     ))
    
#     if 'SMA_50' in data.columns:
#         fig.add_trace(go.Scatter(
#             x=data.index, y=data['SMA_50'].squeeze(),
#             mode='lines', name='SMA 50',
#             line=dict(color='#ff9800', width=2)
#         ))
    
#     if 'EMA_20' in data.columns:
#         fig.add_trace(go.Scatter(
#             x=data.index, y=data['EMA_20'].squeeze(),
#             mode='lines', name='EMA 20',
#             line=dict(color='#2196f3', width=2)
#         ))
    
#     fig.update_layout(
#         title=f'{ticker_symbol} Price History',
#         xaxis_title='Date', yaxis_title='Price (₹)',
#         height=600, template='plotly_white',
#         hovermode='x unified', xaxis_rangeslider_visible=False
#     )
    
#     st.plotly_chart(fig, use_container_width=True)

#     # --- RSI ANALYSIS ---
#     st.subheader("📉 Relative Strength Index (RSI)")
    
#     if 'RSI' in data.columns and not data['RSI'].isna().all():
#         col1, col2, col3 = st.columns(3)
        
#         last_rsi = data['RSI'].iloc[-1]
#         prev_rsi = data['RSI'].iloc[-2] if len(data) > 1 else last_rsi
#         rsi_change = last_rsi - prev_rsi
        
#         with col1:
#             st.metric("Current RSI (14)", f"{last_rsi:.2f}", f"{rsi_change:+.2f}")
        
#         with col2:
#             if last_rsi > 70:
#                 st.error("🔴 Overbought (RSI > 70)")
#                 signal = "Consider Selling"
#             elif last_rsi < 30:
#                 st.success("🟢 Oversold (RSI < 30)")
#                 signal = "Consider Buying"
#             else:
#                 st.info("⚪ Neutral Zone")
#                 signal = "Hold Position"
#             st.write(f"**Signal:** {signal}")
        
#         with col3:
#             rsi_7d_avg = data['RSI'].tail(7).mean()
#             st.metric("7-Day Avg RSI", f"{rsi_7d_avg:.2f}")
        
#         fig_rsi = go.Figure()
#         fig_rsi.add_trace(go.Scatter(
#             x=data.index[-60:], y=data['RSI'].tail(60),
#             mode='lines', name='RSI',
#             line=dict(color='#9c27b0', width=2)
#         ))
        
#         fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
#         fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
#         fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral")
        
#         fig_rsi.update_layout(
#             title='RSI Trend (Last 60 Days)',
#             xaxis_title='Date', yaxis_title='RSI Value',
#             height=300, template='plotly_white',
#             yaxis=dict(range=[0, 100])
#         )
        
#         st.plotly_chart(fig_rsi, use_container_width=True)

#     # --- ADVANCED AI PREDICTION MODELS ---
#     st.divider()
#     st.subheader("🤖 Advanced Deep Learning Price Predictions")
#     st.caption("📌 Using LSTM, Bidirectional LSTM, GRU, TFT, and TSMixer models")
    
#     # Model selection
#     # If TensorFlow failed to import, show a clear message and skip DL UI
#     if not TF_AVAILABLE:
#         st.error("⚠️ TensorFlow is not available in this Python environment. Advanced deep-learning models are disabled.")
#         if TF_IMPORT_ERROR:
#             st.caption("Import error (first 1000 chars):")
#             st.code(TF_IMPORT_ERROR[:1000])

#         # One-click diagnostic for TensorFlow issues
#         with st.expander("Run diagnostic to gather environment details", expanded=False):
#             st.write("This diagnostic checks Python info, TF import (attempt), presence of common MSVC runtime DLLs, NVIDIA/CUDA availability, and relevant PATH entries.")
#             if st.button("Run TF diagnostic"):
#                 import sys, platform, traceback, os, shutil, subprocess

#                 out = []
#                 out.append("--- PYTHON ---")
#                 out.append(f"executable: {sys.executable}")
#                 out.append(f"version: {sys.version}")
#                 out.append(f"platform: {platform.platform()} | machine: {platform.machine()}")

#                 out.append("\n--- TF IMPORT ATTEMPT ---")
#                 try:
#                     import importlib
#                     tf_test = importlib.import_module('tensorflow')
#                     out.append(f"tensorflow import OK: {getattr(tf_test, '__version__', 'unknown')}")
#                 except Exception as err:
#                     out.append("tensorflow import FAILED")
#                     out.append(''.join(traceback.format_exception_only(type(err), err)))
#                     tb = traceback.format_exc()
#                     out.append("Full traceback:\n")
#                     out.append(tb)

#                 out.append("\n--- MSVC RUNTIME DLLS ---")
#                 dlls = [
#                     r"C:\Windows\System32\msvcp140.dll",
#                     r"C:\Windows\System32\vcruntime140.dll",
#                     r"C:\Windows\SysWOW64\msvcp140.dll",
#                     r"C:\Windows\SysWOW64\vcruntime140.dll",
#                 ]
#                 for d in dlls:
#                     out.append(f"{d}: {os.path.exists(d)}")

#                 out.append("\n--- NVIDIA / CUDA ---")
#                 nvidia = shutil.which('nvidia-smi')
#                 if nvidia:
#                     try:
#                         res = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
#                         out.append(res.stdout.strip())
#                     except Exception as e:
#                         out.append(f"nvidia-smi call failed: {e}")
#                 else:
#                     out.append('nvidia-smi not found on PATH')

#                 out.append("\n--- PATH entries with cuda/cudnn ---")
#                 path_hits = [p for p in os.environ.get('PATH','').split(os.pathsep) if p and ('cuda' in p.lower() or 'cudnn' in p.lower())]
#                 out.append(str(path_hits))

#                 st.code('\n'.join(out))

#         st.info("To enable TensorFlow, ensure you're running the app with the same Python interpreter where TensorFlow is installed (e.g. `py -3.12 -m streamlit run app.py`) and install the Microsoft Visual C++ Redistributable if needed. See project README for details.")
#         return

#     model_options = st.multiselect(
#         "Select AI Models to Train:",
#         ["LSTM", "Bidirectional LSTM", "GRU", "TFT (Temporal Fusion Transformer)", "TSMixer"],
#         default=["Bidirectional LSTM", "GRU"]
#     )
    
#     if not model_options:
#         st.info("👆 Please select at least one model to generate predictions")
#         return
    
#     # Prepare data
#     with st.spinner('Preparing data for deep learning models...'):
#         # Create features
#         df_ml = data[['Close', 'Open', 'High', 'Low', 'Volume']].copy()
#         df_ml = df_ml.fillna(method='ffill').fillna(method='bfill')
        
#         # Scale data
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         scaled_data = scaler.fit_transform(df_ml)
        
#         # Prepare sequences
#         n_steps = 60  # Use 60 days to predict next day
#         X, y = prepare_sequences(scaled_data[:, 0], n_steps)
        
#         # Reshape for multiple features
#         X_multi = []
#         for i in range(n_steps, len(scaled_data)):
#             X_multi.append(scaled_data[i-n_steps:i])
#         X_multi = np.array(X_multi)
#         y = scaled_data[n_steps:, 0]
        
#         # Split data
#         split_idx = int(len(X_multi) * 0.8)
#         X_train, X_test = X_multi[:split_idx], X_multi[split_idx:]
#         y_train, y_test = y[:split_idx], y[split_idx:]
        
#         # Prepare scaler for inverse transform
#         close_scaler = MinMaxScaler(feature_range=(0, 1))
#         close_scaler.fit(df_ml[['Close']])
    
#     st.success(f"✅ Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
    
#     # Train selected models
#     predictions_dict = {}
#     metrics_dict = {}
    
#     for model_name in model_options:
#         st.markdown(f"### 🔬 {model_name} Model")
        
#         try:
#             # Build model
#             if model_name == "LSTM":
#                 model = build_lstm_model(n_steps, X_multi.shape[2])
#             elif model_name == "Bidirectional LSTM":
#                 model = build_bidirectional_lstm_model(n_steps, X_multi.shape[2])
#             elif model_name == "GRU":
#                 model = build_gru_model(n_steps, X_multi.shape[2])
#             elif model_name == "TFT (Temporal Fusion Transformer)":
#                 model = build_tft_model(n_steps, X_multi.shape[2])
#             elif model_name == "TSMixer":
#                 model = build_tsmixer_model(n_steps, X_multi.shape[2])
            
#             # Train and predict
#             test_predictions, history = train_and_predict(
#                 model, X_train, y_train, X_test, close_scaler, model_name
#             )
            
#             # Get actual test values
#             y_test_actual = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
#             # Calculate metrics
#             metrics = calculate_metrics(y_test_actual, test_predictions)
#             metrics_dict[model_name] = metrics
            
#             # Display metrics
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.metric("MAE", f"₹{metrics['MAE']:.2f}")
#             with col2:
#                 st.metric("RMSE", f"₹{metrics['RMSE']:.2f}")
#             with col3:
#                 st.metric("R² Score", f"{metrics['R²']:.4f}")
#             with col4:
#                 st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            
#             # Future predictions
#             last_sequence = scaled_data[-n_steps:]
#             future_predictions = []
            
#             for _ in range(7):  # Predict 7 days
#                 last_sequence_reshaped = last_sequence.reshape(1, n_steps, X_multi.shape[2])
#                 next_pred = model.predict(last_sequence_reshaped, verbose=0)
                
#                 # Create full feature vector for next prediction
#                 next_full = np.zeros((1, X_multi.shape[2]))
#                 next_full[0, 0] = next_pred[0, 0]
                
#                 last_sequence = np.vstack([last_sequence[1:], next_full])
#                 future_predictions.append(next_pred[0, 0])
            
#             future_predictions = close_scaler.inverse_transform(
#                 np.array(future_predictions).reshape(-1, 1)
#             ).flatten()
            
#             predictions_dict[model_name] = {
#                 'test': test_predictions,
#                 'future': future_predictions,
#                 'history': history
#             }
            
#             st.success(f"✅ {model_name} trained successfully!")
            
#         except Exception as e:
#             st.error(f"❌ Error training {model_name}: {str(e)}")
#             continue
    
#     # --- COMPARISON CHARTS ---
#     if predictions_dict:
#         st.divider()
#         st.subheader("📊 Model Comparison & Predictions")
        
#         # Test predictions comparison
#         st.markdown("#### 🔍 Test Set Performance Comparison")
        
#         fig_test = go.Figure()
        
#         # Add actual values
#         test_dates = data.index[split_idx + n_steps:]
#         fig_test.add_trace(go.Scatter(
#             x=test_dates,
#             y=y_test_actual,
#             mode='lines',
#             name='Actual Price',
#             line=dict(color='black', width=3)
#         ))
        
#         # Add predictions from each model
#         colors = ['#1976d2', '#f57c00', '#388e3c', '#d32f2f', '#7b1fa2']
#         for idx, (model_name, preds) in enumerate(predictions_dict.items()):
#             fig_test.add_trace(go.Scatter(
#                 x=test_dates,
#                 y=preds['test'],
#                 mode='lines',
#                 name=model_name,
#                 line=dict(color=colors[idx % len(colors)], width=2, dash='dash')
#             ))
        
#         fig_test.update_layout(
#             title='Model Predictions vs Actual Prices (Test Set)',
#             xaxis_title='Date',
#             yaxis_title='Price (₹)',
#             height=500,
#             template='plotly_white',
#             hovermode='x unified'
#         )
        
#         st.plotly_chart(fig_test, use_container_width=True)
        
#         # Future predictions comparison
#         st.markdown("#### 🔮 7-Day Future Predictions Comparison")
        
#         last_date = data.index[-1]
#         future_dates = get_trading_days(last_date, num_days=7)
        
#         fig_future = go.Figure()
        
#         # Add historical data (last 30 days)
#         historical_last_30 = data['Close'].tail(30)
#         fig_future.add_trace(go.Scatter(
#             x=historical_last_30.index,
#             y=historical_last_30.values,
#             mode='lines',
#             name='Historical Price',
#             line=dict(color='black', width=3)
#         ))
        
#         # Add future predictions from each model
#         for idx, (model_name, preds) in enumerate(predictions_dict.items()):
#             fig_future.add_trace(go.Scatter(
#                 x=future_dates,
#                 y=preds['future'],
#                 mode='lines+markers',
#                 name=f'{model_name} Prediction',
#                 line=dict(color=colors[idx % len(colors)], width=3, dash='dash'),
#                 marker=dict(size=10)
#             ))
        
#         fig_future.update_layout(
#             title='7-Day Future Price Predictions by All Models',
#             xaxis_title='Date',
#             yaxis_title='Price (₹)',
#             height=500,
#             template='plotly_white',
#             hovermode='x unified'
#         )
        
#         st.plotly_chart(fig_future, use_container_width=True)
        
#         # --- ENSEMBLE PREDICTION ---
#         st.divider()
#         st.subheader("🎯 Ensemble Prediction (Average of All Models)")
        
#         # Calculate ensemble average
#         ensemble_future = np.mean([preds['future'] for preds in predictions_dict.values()], axis=0)
        
#         # Create prediction table
#         pred_df = pd.DataFrame({
#             'Date': future_dates,
#             'Day': [d.strftime('%A') for d in future_dates],
#             'Ensemble Prediction': ensemble_future
#         })
        
#         # Add individual model predictions
#         for model_name, preds in predictions_dict.items():
#             pred_df[model_name] = preds['future']
        
#         col1, col2 = st.columns([2, 1])
        
#         with col1:
#             st.write("### 📅 7-Day Ensemble Forecast")
            
#             # Create styled HTML table
#             min_price = ensemble_future.min()
#             max_price = ensemble_future.max()
            
#             html_table = "<table style='width:100%; border-collapse: collapse;'>"
#             html_table += "<thead><tr style='background-color: #f0f0f0;'>"
#             html_table += "<th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Date</th>"
#             html_table += "<th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Day</th>"
#             html_table += "<th style='padding: 10px; text-align: right; border: 1px solid #ddd;'>Ensemble Price</th>"
#             html_table += "</tr></thead><tbody>"
            
#             for idx, row in pred_df.iterrows():
#                 bg_color = get_color_for_price(row['Ensemble Prediction'], min_price, max_price)
#                 html_table += f"<tr style='background-color: {bg_color};'>"
#                 html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Date'].strftime('%Y-%m-%d')}</td>"
#                 html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Day']}</td>"
#                 html_table += f"<td style='padding: 10px; text-align: right; border: 1px solid #ddd; font-weight: bold;'>₹ {row['Ensemble Prediction']:.2f}</td>"
#                 html_table += "</tr>"
            
#             html_table += "</tbody></table>"
#             st.markdown(html_table, unsafe_allow_html=True)
        
#         with col2:
#             current_price = data['Close'].iloc[-1]
#             predicted_price = ensemble_future[-1]
#             change = ((predicted_price - current_price) / current_price) * 100
            
#             st.metric(
#                 "7-Day Target Price",
#                 f"₹ {predicted_price:.2f}",
#                 f"{change:+.2f}%",
#                 delta_color="normal"
#             )
            
#             # Calculate average R² across models
#             avg_r2 = np.mean([m['R²'] for m in metrics_dict.values()])
#             st.metric("Average Model R²", f"{avg_r2:.4f}")
            
#             if change > 2:
#                 st.success("📈 **Bullish Trend**\n\nConsider: BUY/HOLD")
#             elif change < -2:
#                 st.error("📉 **Bearish Trend**\n\nConsider: SELL/CAUTION")
#             else:
#                 st.info("➡️ **Sideways Trend**\n\nConsider: HOLD")
        
#         # Model performance comparison
#         st.divider()
#         st.subheader("📊 Model Performance Metrics Comparison")
        
#         metrics_df = pd.DataFrame(metrics_dict).T
#         st.dataframe(
#             metrics_df.style.format({
#                 'MAE': '₹{:.2f}',
#                 'RMSE': '₹{:.2f}',
#                 'R²': '{:.4f}',
#                 'MAPE': '{:.2f}%'
#             }).background_gradient(cmap='RdYlGn_r', subset=['MAE', 'RMSE', 'MAPE'])
#               .background_gradient(cmap='RdYlGn', subset=['R²']),
#             use_container_width=True
#         )
        
#         # Best model
#         best_model = metrics_df['R²'].idxmax()
#         st.success(f"🏆 **Best Performing Model:** {best_model} (R² = {metrics_df.loc[best_model, 'R²']:.4f})")
    
#     # Disclaimer
#     st.divider()
#     st.info("""
#     ⚠️ **Disclaimer:** These predictions are generated using advanced deep learning models for educational purposes only. 
#     - Models include: LSTM, Bidirectional LSTM, GRU, Temporal Fusion Transformer, and TSMixer
#     - Past performance does not guarantee future results
#     - Financial markets are influenced by many unpredictable factors
#     - Always do your own research and consult with financial advisors before investing
#     - Use these predictions as one of many tools in your investment analysis
#     """)



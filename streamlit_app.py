# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# # ------------------------------
# # App Title
# # ------------------------------
# st.title("📊 Gold-Denominated Turkish Stocks")
# st.markdown("""
# **How it works:** This analysis compares Turkish stock performance against physical gold (IAU ETF) converted to Turkish Lira. 
# The gold-denominated ratio is calculated by dividing the stock price by the TRY-adjusted gold price (IAU × USD/TRY exchange rate). 
# Both series are normalized to start at 1.0, making it easy to compare relative performance over time. A rising ratio indicates the stock 
# is outperforming gold, while a falling ratio suggests gold is the better investment during that period.
# """)
# # ------------------------------
# # User Input
# # ------------------------------
# ticker = st.text_input("Enter Borsa Istanbul stock ticker (e.g., SAHOL.IS):", "SAHOL.IS")
# years = st.number_input("Enter number of years for analysis:", min_value=1, max_value=20, value=2, step=1)
# ticker_gold = "IAU"
# ticker_xrate = "TRY=X"
# period = f"{years}y"  # dynamic period
# if st.button("Run Analysis"):
#     # ------------------------------
#     # Download Data
#     # ------------------------------
#     df = yf.download([ticker, ticker_gold, ticker_xrate], period=period, group_by="ticker")
#     df = df.dropna()
#     close_stock = df[ticker]["Close"]
#     close_gold = df[ticker_gold]["Close"]
#     close_xrate = df[ticker_xrate]["Close"]
#     data = pd.DataFrame({
#         "Stock": close_stock,
#         "Gold": close_gold
#     }).dropna()
#     data["Gold"] = (close_xrate * data["Gold"])
#     data["Gold"] = data["Gold"] / data["Gold"].iloc[0]
#     data["Stock"] = data["Stock"] / data["Stock"].iloc[0]
#     # Gold-denominated stock price
#     data["Ratio"] = data["Stock"] / data["Gold"]
#     # Moving averages
#     data["MA9"] = data["Ratio"].rolling(4).mean()
#     data["MA50"] = data["Ratio"].rolling(50).mean()
#     # Bollinger Bands
#     data["BB_Mid"] = data["Ratio"].rolling(20).mean()
#     data["BB_Upper"] = data["BB_Mid"] + 2 * data["Ratio"].rolling(20).std()
#     data["BB_Lower"] = data["BB_Mid"] - 2 * data["Ratio"].rolling(20).std()
#     # RSI
#     def compute_rsi(series, period=22):
#         delta = series.diff()
#         gain = delta.where(delta > 0, 0.0)
#         loss = -delta.where(delta < 0, 0.0)
#         avg_gain = gain.rolling(window=period).mean()
#         avg_loss = loss.rolling(window=period).mean()
#         rs = avg_gain / avg_loss
#         rsi = 100 - (100 / (1 + rs))
#         return rsi
#     data["RSI"] = compute_rsi(data["Ratio"], 50)
#     # Z-Score
#     window = 60
#     mean = data["Ratio"].rolling(window).mean()
#     std = data["Ratio"].rolling(window).std()
#     data["ZScore"] = (data["Ratio"] - mean) / std
#     # ------------------------------
#     # Latest Values Table
#     # ------------------------------
#     st.subheader("📋 Latest Values")
    
#     latest_date = data.index[-1].strftime('%Y-%m-%d')
#     latest_stock_close = close_stock.iloc[-1]
#     latest_gold_denominated = data["Ratio"].iloc[-1]
#     latest_rsi = data["RSI"].iloc[-1]
#     latest_zscore = data["ZScore"].iloc[-1]
    
#     summary_data = {
#         "Metric": ["Date", "Stock Close Price (TRY)", "Gold-Denominated Ratio", "RSI (50)", "Z-Score (60d)"],
#         "Value": [
#             latest_date,
#             f"{latest_stock_close:.2f}",
#             f"{latest_gold_denominated:.4f}",
#             f"{latest_rsi:.2f}" if not np.isnan(latest_rsi) else "N/A",
#             f"{latest_zscore:.2f}" if not np.isnan(latest_zscore) else "N/A"
#         ]
#     }
    
#     summary_df = pd.DataFrame(summary_data)
#     st.table(summary_df)
    
#     # ------------------------------
#     # Plotting
#     # ------------------------------
#     fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
#     # 1. Ratio plot
#     overall_avg = data["Ratio"].mean()
#     axes[0].plot(data.index, data["Ratio"], label=ticker + "/ Gold", color="blue")
#     axes[0].plot(data.index, data["MA9"], label="MA 9d", color="orange", linestyle="--")
#     axes[0].plot(data.index, data["MA50"], label="MA 50d", color="magenta", linestyle="--")
#     axes[0].fill_between(data.index, data["BB_Upper"], data["BB_Lower"], color="gray", alpha=0.2, label="Bollinger Bands")
#     axes[0].axhline(overall_avg, color="black", linestyle="--", label=f"Overall Avg ({overall_avg:.2f})")
#     axes[0].set_title("Gold-Denominated " + ticker)
#     axes[0].legend()
#     axes[0].grid(axis='y', linestyle='--', alpha=0.5)
#     # 2. RSI
#     axes[1].plot(data.index, data["RSI"], label="RSI (50)", color="purple")
#     axes[1].axhline(30, color="green", linestyle="--", label="Buy Level (30)")
#     axes[1].axhline(70, color="red", linestyle="--", label="Sell Level (70)")
#     axes[1].set_title("RSI")
#     axes[1].legend()
#     # 3. Z-Score
#     axes[2].plot(data.index, data["ZScore"], label="Z-Score (60d)", color="black")
#     axes[2].axhline(-2, color="green", linestyle="--", label="Standard -2")
#     axes[2].axhline(2, color="red", linestyle="--", label="Standard +2")
#     axes[2].set_title("Z-Score")
#     axes[2].legend()
#     date_formatter = mdates.ConciseDateFormatter(mdates.AutoDateLocator())
#     for ax in axes:
#         ax.xaxis.set_major_formatter(date_formatter)
#         ax.tick_params(axis='x', rotation=45, labelbottom=True)
#     st.pyplot(fig)



import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# App Title
# ------------------------------
st.title("📊 Gold-Denominated Turkish Stocks")
st.markdown("""
**How it works:** This analysis compares Turkish stock performance against physical gold (IAU ETF) converted to Turkish Lira. 
The gold-denominated ratio is calculated by dividing the stock price by the TRY-adjusted gold price (IAU × USD/TRY exchange rate). 
Both series are normalized to start at 1.0, making it easy to compare relative performance over time. A rising ratio indicates the stock 
is outperforming gold, while a falling ratio suggests gold is the better investment during that period.
""")
# ------------------------------
# User Input
# ------------------------------
ticker = st.text_input("Enter Borsa Istanbul stock ticker (e.g., SAHOL.IS):", "SAHOL.IS")
years = st.number_input("Enter number of years for analysis:", min_value=1, max_value=20, value=2, step=1)
ticker_gold = "IAU"
ticker_xrate = "TRY=X"
period = f"{years}y"  # dynamic period

if st.button("Run Analysis"):
    # ------------------------------
    # Download Data
    # ------------------------------
    with st.spinner("Downloading data..."):
        df = yf.download([ticker, ticker_gold, ticker_xrate], period=period, group_by="ticker")
        df = df.dropna()
        close_stock = df[ticker]["Close"]
        close_gold = df[ticker_gold]["Close"]
        close_xrate = df[ticker_xrate]["Close"]
        
        data = pd.DataFrame({
            "Stock": close_stock,
            "Gold": close_gold
        }).dropna()
        data["Gold"] = (close_xrate * data["Gold"])
        data["Gold"] = data["Gold"] / data["Gold"].iloc[0]
        data["Stock"] = data["Stock"] / data["Stock"].iloc[0]
        
        # Gold-denominated stock price
        data["Ratio"] = data["Stock"] / data["Gold"]
        
        # Moving averages
        data["MA9"] = data["Ratio"].rolling(4).mean()
        data["MA50"] = data["Ratio"].rolling(50).mean()
        
        # Bollinger Bands
        data["BB_Mid"] = data["Ratio"].rolling(20).mean()
        data["BB_Upper"] = data["BB_Mid"] + 2 * data["Ratio"].rolling(20).std()
        data["BB_Lower"] = data["BB_Mid"] - 2 * data["Ratio"].rolling(20).std()
        
        # RSI
        def compute_rsi(series, period=22):
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        data["RSI"] = compute_rsi(data["Ratio"], 50)
        
        # Z-Score
        window = 60
        mean = data["Ratio"].rolling(window).mean()
        std = data["Ratio"].rolling(window).std()
        data["ZScore"] = (data["Ratio"] - mean) / std
    
    # ------------------------------
    # LSTM Forecasting
    # ------------------------------
    with st.spinner("Training LSTM model for forecasting..."):
        # Prepare data for LSTM
        ratio_values = data["Ratio"].values
        
        # Calculate differences (deltas) for training
        ratio_diff = np.diff(ratio_values)
        ratio_diff = ratio_diff.reshape(-1, 1)
        
        # Scale the differences
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_diff = scaler.fit_transform(ratio_diff)
        
        # Create sequences
        lookback = 10
        forecast_horizon = 10
        
        X, y = [], []
        for i in range(lookback, len(scaled_diff)):
            X.append(scaled_diff[i-lookback:i, 0])
            y.append(scaled_diff[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # Train the model
        model.fit(X, y, batch_size=32, epochs=50, verbose=0)
        
        # Make predictions for next 10 days
        last_sequence = scaled_diff[-lookback:]
        predictions = []
        
        # Start from the last actual value
        current_value = ratio_values[-1]
        current_sequence = last_sequence.copy()
        
        for _ in range(forecast_horizon):
            current_sequence_reshaped = current_sequence.reshape(1, lookback, 1)
            next_diff_scaled = model.predict(current_sequence_reshaped, verbose=0)
            
            # Inverse transform the predicted difference
            next_diff = scaler.inverse_transform(next_diff_scaled.reshape(-1, 1))[0, 0]
            
            # Add the difference to get the next value
            current_value = current_value + next_diff
            predictions.append(current_value)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], next_diff_scaled, axis=0)
        
        predictions = np.array(predictions)
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Ratio': predictions.flatten()
        })
        forecast_df.set_index('Date', inplace=True)
    
    # ------------------------------
    # Latest Values Table
    # ------------------------------
    st.subheader("📋 Latest Values")
    
    latest_date = data.index[-1].strftime('%Y-%m-%d')
    latest_stock_close = close_stock.iloc[-1]
    latest_gold_denominated = data["Ratio"].iloc[-1]
    latest_rsi = data["RSI"].iloc[-1]
    latest_zscore = data["ZScore"].iloc[-1]
    
    summary_data = {
        "Metric": ["Date", "Stock Close Price (TRY)", "Gold-Denominated Ratio", "RSI (50)", "Z-Score (60d)"],
        "Value": [
            latest_date,
            f"{latest_stock_close:.2f}",
            f"{latest_gold_denominated:.4f}",
            f"{latest_rsi:.2f}" if not np.isnan(latest_rsi) else "N/A",
            f"{latest_zscore:.2f}" if not np.isnan(latest_zscore) else "N/A"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)
    
    # ------------------------------
    # Plotting
    # ------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=False)
    
    # 1. Ratio plot with forecast
    overall_avg = data["Ratio"].mean()
    axes[0].plot(data.index, data["Ratio"], label=ticker + " / Gold", color="blue")
    axes[0].plot(data.index, data["MA9"], label="MA 9d", color="orange", linestyle="--")
    axes[0].plot(data.index, data["MA50"], label="MA 50d", color="magenta", linestyle="--")
    axes[0].fill_between(data.index, data["BB_Upper"], data["BB_Lower"], color="gray", alpha=0.2, label="Bollinger Bands")
    axes[0].axhline(overall_avg, color="black", linestyle="--", label=f"Overall Avg ({overall_avg:.2f})")
    
    # Add forecast to the plot
    axes[0].plot(forecast_df.index, forecast_df["Predicted_Ratio"], label="10-Day LSTM Forecast", 
                 color="red", linestyle="-", linewidth=2, marker='o', markersize=4)
    axes[0].axvline(x=data.index[-1], color='green', linestyle=':', linewidth=1.5, label='Forecast Start')
    
    axes[0].set_title("Gold-Denominated " + ticker + " with 10-Day LSTM Forecast")
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # 2. RSI
    axes[1].plot(data.index, data["RSI"], label="RSI (50)", color="purple")
    axes[1].axhline(30, color="green", linestyle="--", label="Buy Level (30)")
    axes[1].axhline(70, color="red", linestyle="--", label="Sell Level (70)")
    axes[1].set_title("RSI")
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    # 3. Z-Score
    axes[2].plot(data.index, data["ZScore"], label="Z-Score (60d)", color="black")
    axes[2].axhline(-2, color="green", linestyle="--", label="Standard -2")
    axes[2].axhline(2, color="red", linestyle="--", label="Standard +2")
    axes[2].set_title("Z-Score")
    axes[2].legend()
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    
    # 4. Forecast Detail Plot
    # Show last 50 days of historical data + forecast
    last_50_days = data.tail(50)
    axes[3].plot(last_50_days.index, last_50_days["Ratio"], label="Historical Ratio (Last 50 Days)", 
                 color="blue", linewidth=2)
    axes[3].plot(forecast_df.index, forecast_df["Predicted_Ratio"], label="10-Day LSTM Forecast", 
                 color="red", linestyle="-", linewidth=2, marker='o', markersize=5)
    axes[3].axvline(x=data.index[-1], color='green', linestyle=':', linewidth=2, label='Forecast Start')
    axes[3].set_title("Detailed View: Last 50 Days + 10-Day LSTM Forecast")
    axes[3].legend()
    axes[3].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Format dates for all subplots
    for i, ax in enumerate(axes):
        date_formatter = mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(date_formatter)
        ax.tick_params(axis='x', rotation=45, labelbottom=True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ------------------------------
    # Forecast Table
    # ------------------------------
    st.subheader("📈 10-Day LSTM Forecast")
    forecast_display = forecast_df.copy()
    forecast_display['Date'] = forecast_display.index.strftime('%Y-%m-%d')
    forecast_display['Predicted Ratio'] = forecast_display['Predicted_Ratio'].apply(lambda x: f"{x:.4f}")
    forecast_display = forecast_display[['Date', 'Predicted Ratio']].reset_index(drop=True)
    st.table(forecast_display)
    
    st.success("✅ Analysis complete!")


        # ------------------------------
    # Methodology Explanation
    # ------------------------------
    st.subheader("📚 About the LSTM Forecasting Model")
    
    st.markdown("""
    **What is LSTM?**
    
    Long Short-Term Memory (LSTM) is a type of neural network specifically designed to learn patterns from sequential data, 
    making it ideal for time series forecasting. Unlike traditional models, LSTMs can remember long-term dependencies in the data, 
    which helps capture complex trends and patterns in stock price movements.
    
    **How Our Model Works:**
    
    1. **Data Preparation:** We use the historical gold-denominated ratio values (your stock price divided by gold price in TRY) as our time series data.
    
    2. **Differential Learning:** Instead of predicting absolute values, the model learns to predict *changes* (deltas) between consecutive days. 
       This approach ensures smooth transitions from historical data to forecasts and better captures the momentum of price movements.
    
    3. **Lookback Window:** The model uses the last 10 days of data to predict the next day's change. This 10-day window provides enough 
       context to identify short-term trends without being influenced by very old patterns.
    
    4. **Sequential Forecasting:** To predict 10 days ahead, the model makes one prediction at a time, then uses that prediction as part of 
       the input for the next prediction. This recursive approach builds the forecast step-by-step.
    
    5. **Model Architecture:**
       - Two LSTM layers (50 units each) to capture complex patterns
       - Dropout layers (20%) to prevent overfitting
       - Dense layers for final prediction output
       - Trained on all available historical data (50 epochs)
    
    **Important Limitations:**
    
    - ⚠️ **Not Financial Advice:** This model is for educational and analytical purposes only. It cannot predict unexpected market events, 
      news, or fundamental changes.
    - ⚠️ **Uncertainty Increases:** Predictions become less reliable the further into the future they extend. The 10th day forecast is 
      much less certain than the 1st day.
    - ⚠️ **Historical Patterns:** The model assumes future patterns will resemble historical behavior. Market regime changes can make 
      predictions inaccurate.
    
    **How to Interpret:**
    
    Use the forecast as one of many tools in your analysis. Compare it with the RSI and Z-Score indicators to get a more complete picture. 
    If multiple indicators align (e.g., RSI suggests oversold conditions AND the forecast shows upward movement), the signal may be stronger.
    """)
    
    st.success("✅ Analysis complete!")

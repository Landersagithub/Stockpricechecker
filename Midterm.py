import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# Use Streamlit secrets or environment variable for API key
# For local testing, you can hardcode temporarily, but use secrets for deployment
try:
    API_KEY = st.secrets["ALPHA_VANTAGE_KEY"]
except:
    API_KEY = "DWHS823B2TK98J5Z"  # Replace with your new key

st.set_page_config(page_title="Stock Price App", layout="wide")

st.title("📈 Simple Stock Price App")

# Add info about API limits
st.info("💡 Alpha Vantage free tier: 5 calls/minute, 500/day. Data fetches once per symbol entry.")

symbol = st.text_input("Enter stock symbol (e.g., AAPL, TSLA, MSFT):")

# Buttons - Row 1
st.write("### Actions:")
col1, col2, col3 = st.columns(3)
get_latest_btn = col1.button("📌 Get Latest Price", use_container_width=True)
history_btn = col2.button("📜 Show Historical Data", use_container_width=True)
chart_btn = col3.button("📊 Show Chart", use_container_width=True)

# Button - Row 2
predict_btn = st.button("🔮 Predict Future Prices", use_container_width=True, type="primary")

@st.cache_data(ttl=3600)  # Cache for 1 hour to avoid repeated API calls
def fetch_data(symbol):
    """Fetch stock data from Alpha Vantage API"""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return {}

if symbol:
    with st.spinner(f"Fetching data for {symbol.upper()}..."):
        data = fetch_data(symbol)

    if "Time Series (Daily)" in data:
        # Convert historical data to DataFrame
        hist_df = pd.DataFrame(data["Time Series (Daily)"]).T
        hist_df.index = pd.to_datetime(hist_df.index)
        hist_df.index.name = "Date"
        hist_df.columns = ["Open", "High", "Low", "Close", "Volume"]

        # Convert strings to floats
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            hist_df[col] = hist_df[col].astype(float)

        # Sort by date (newest first)
        hist_df = hist_df.sort_index(ascending=False)

        # ======================
        # 📌 LATEST PRICE BUTTON
        # ======================
        if get_latest_btn:
            st.subheader("📌 Latest Stock Price")

            latest_date = hist_df.index[0]
            latest_row = hist_df.loc[latest_date]

            # Calculate daily change
            if len(hist_df) > 1:
                prev_close = hist_df.iloc[1]["Close"]
                change = latest_row["Close"] - prev_close
                change_pct = (change / prev_close) * 100
                change_color = "green" if change >= 0 else "red"
                change_symbol = "▲" if change >= 0 else "▼"
            else:
                change = 0
                change_pct = 0
                change_color = "gray"
                change_symbol = "●"

            st.markdown(
                f"""
                <div style="
                    padding: 20px;
                    background-color: #f7f7f7;
                    border-radius: 10px;
                    border: 1px solid #ddd;
                ">
                    <h3 style="margin-top:0;">📅 Latest data for <b>{symbol.upper()}</b></h3>
                    <p style="font-size: 14px; color: #666;">Date: {latest_date.strftime('%Y-%m-%d')}</p>
                    <p style="font-size: 24px; font-weight: bold; color: {change_color};">
                        ${latest_row["Close"]:.2f} 
                        <span style="font-size: 18px;">{change_symbol} {change:.2f} ({change_pct:+.2f}%)</span>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            latest_df = pd.DataFrame({
                "Metric": ["Open", "High", "Low", "Close", "Volume"],
                "Value": [
                    f"${latest_row['Open']:.2f}",
                    f"${latest_row['High']:.2f}",
                    f"${latest_row['Low']:.2f}",
                    f"${latest_row['Close']:.2f}",
                    f"{int(latest_row['Volume']):,}"
                ]
            })

            st.table(latest_df)

        # ======================
        # 📜 HISTORICAL BUTTON
        # ======================
        if history_btn:
            st.subheader("📜 Historical Stock Prices")
            
            # Add date range selector
            days = st.selectbox("Select time period:", [30, 60, 90, 180, 365, "All"], index=1)
            
            if days == "All":
                display_df = hist_df
            else:
                display_df = hist_df.head(days)
            
            st.dataframe(display_df.style.format({
                "Open": "${:.2f}",
                "High": "${:.2f}",
                "Low": "${:.2f}",
                "Close": "${:.2f}",
                "Volume": "{:,.0f}"
            }), use_container_width=True)
            
            # Download button
            csv = display_df.to_csv()
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{symbol}_stock_data.csv",
                mime="text/csv"
            )

        # ======================
        # 📊 CHART BUTTON
        # ======================
        if chart_btn:
            st.subheader("📊 Stock Charts")

            # Date range selector for charts
            days = st.selectbox("Chart time period:", [30, 60, 90, 180, 365, "All"], index=1, key="chart_days")
            
            if days == "All":
                chart_df = hist_df.sort_index(ascending=True)
            else:
                chart_df = hist_df.head(days).sort_index(ascending=True)

            # ---------- 1. Line Chart ----------
            st.write("### 📉 Closing Price Trend")
            st.line_chart(chart_df["Close"])

            # ---------- 2. Candlestick Chart ----------
            st.write("### 🕯️ Candlestick Chart")
            candle_fig = go.Figure(data=[go.Candlestick(
                x=chart_df.index,
                open=chart_df["Open"],
                high=chart_df["High"],
                low=chart_df["Low"],
                close=chart_df["Close"],
                name=symbol.upper()
            )])
            candle_fig.update_layout(
                height=600,
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode="x unified"
            )
            st.plotly_chart(candle_fig, use_container_width=True)

            # ---------- 3. Volume Chart ----------
            st.write("### 📊 Volume Chart")
            st.bar_chart(chart_df["Volume"])

        # ======================
        # 🔮 PREDICT BUTTON
        # ======================
        if predict_btn:
            st.subheader("🔮 Stock Price Prediction (Linear Regression)")
            
            st.warning("⚠️ **Disclaimer:** This is a simple linear regression model for educational purposes only. Real stock prediction requires advanced models and should not be used for actual trading decisions.")
            
            # User inputs for prediction
            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                training_days = st.slider("Training period (days):", 30, 365, 90)
            with col_pred2:
                predict_days = st.slider("Predict ahead (days):", 1, 90, 30)
            
            # Prepare data for prediction
            pred_df = hist_df.sort_index(ascending=True).tail(training_days).copy()
            
            # Create numerical day index
            pred_df['Day_Num'] = np.arange(len(pred_df))
            
            # Train the model
            X = pred_df['Day_Num'].values.reshape(-1, 1)
            y = pred_df['Close'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Make predictions for existing data
            pred_df['Predicted'] = model.predict(X)
            
            # Predict future prices
            future_days = np.arange(len(pred_df), len(pred_df) + predict_days).reshape(-1, 1)
            future_prices = model.predict(future_days)
            
            # Create future dates
            last_date = pred_df.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=predict_days, freq='D')
            
            # Create future dataframe
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Price': future_prices
            })
            future_df.set_index('Date', inplace=True)
            
            # Calculate R² score
            from sklearn.metrics import r2_score, mean_absolute_error
            r2 = r2_score(y, pred_df['Predicted'])
            mae = mean_absolute_error(y, pred_df['Predicted'])
            
            # Display metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("R² Score", f"{r2:.4f}")
            with col_m2:
                st.metric("Mean Abs Error", f"${mae:.2f}")
            with col_m3:
                st.metric("Current Price", f"${pred_df['Close'].iloc[-1]:.2f}")
            with col_m4:
                predicted_price = future_prices[-1]
                price_change = predicted_price - pred_df['Close'].iloc[-1]
                st.metric(f"Predicted ({predict_days}d)", f"${predicted_price:.2f}", f"{price_change:+.2f}")
            
            # Show interpretation
            if r2 > 0.7:
                st.success(f"✅ Strong linear trend detected (R² = {r2:.4f})")
            elif r2 > 0.4:
                st.info(f"ℹ️ Moderate linear trend (R² = {r2:.4f})")
            else:
                st.warning(f"⚠️ Weak linear trend (R² = {r2:.4f}) - predictions may be unreliable")
            
            # ---------- Prediction Table (MOVED UP) ----------
            st.write("### 📋 Future Price Predictions Table")
            
            # Format the future predictions table
            future_display = future_df.copy()
            future_display['Predicted_Price'] = future_display['Predicted_Price'].apply(lambda x: f"${x:.2f}")
            future_display.index = future_display.index.strftime('%Y-%m-%d')
            future_display.index.name = 'Date'
            
            # Show in a nice formatted table
            st.dataframe(future_display, use_container_width=True, height=400)
            
            # Download predictions
            csv_pred = future_df.to_csv()
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_pred,
                file_name=f"{symbol}_predictions.csv",
                mime="text/csv"
            )
            
            st.write("---")  # Divider
            
            # ---------- Prediction Chart ----------
            st.write("### 📈 Price Prediction Chart")
            
            fig_pred = go.Figure()
            
            # Historical actual prices
            fig_pred.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Close'],
                mode='lines',
                name='Actual Price',
                line=dict(color='blue', width=2)
            ))
            
            # Fitted line on historical data
            fig_pred.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Predicted'],
                mode='lines',
                name='Fitted Line',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            # Future predictions
            fig_pred.add_trace(go.Scatter(
                x=future_df.index,
                y=future_df['Predicted_Price'],
                mode='lines+markers',
                name='Future Prediction',
                line=dict(color='red', width=2, dash='dot'),
                marker=dict(size=6)
            ))
            
            fig_pred.update_layout(
                height=600,
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode="x unified",
                showlegend=True
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # ---------- Model Details ----------
            st.write("### 📐 Model Details")
            
            slope = model.coef_[0]
            intercept = model.intercept_
            st.code(f"Price = {intercept:.2f} + {slope:.2f} × Day")
            
            trend_direction = "upward 📈" if slope > 0 else "downward 📉"
            st.write(f"**Trend:** The model shows an {trend_direction} trend of **${abs(slope):.2f} per day**.")

    elif "Note" in data:
        st.warning("⚠️ API rate limit reached. Alpha Vantage allows 5 calls/minute. Please wait and try again.")
    elif "Error Message" in data:
        st.error(f"❌ Invalid stock symbol: {symbol.upper()}")
    else:
        st.error("❌ Unable to fetch data. Please check your API key and try again.")
else:
    st.write("👆 Enter a stock symbol above to get started!")
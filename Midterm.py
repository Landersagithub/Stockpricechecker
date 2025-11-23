import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta, datetime
import requests

# MUST BE FIRST
st.set_page_config(page_title="Stock Price App", layout="wide")

# Initialize ALL session states at the very beginning
if 'show_latest' not in st.session_state:
    st.session_state.show_latest = False
if 'show_history' not in st.session_state:
    st.session_state.show_history = False
if 'show_chart' not in st.session_state:
    st.session_state.show_chart = False
if 'show_predict' not in st.session_state:
    st.session_state.show_predict = False
if 'show_buy' not in st.session_state:
    st.session_state.show_buy = False
if 'show_sell' not in st.session_state:
    st.session_state.show_sell = False

# Exchange rate USD to PHP
USD_TO_PHP = 58.79

st.title("Stock Price Analysis Application")
st.info("🌍 International Stocks: Yahoo Finance | 🇵🇭 Philippine Stocks: Yahoo Finance with .PS suffix")

# Helper text
st.caption("**International stocks:** AAPL, TSLA, MSFT | **Philippine stocks:** Add .PS suffix (JFC.PS, SM.PS, BDO.PS, ALI.PS)")

symbol = st.text_input("Enter stock symbol:")

# Buttons - Row 1
st.write("### Actions:")
col1, col2, col3 = st.columns(3)
if col1.button("Get Latest Price", use_container_width=True):
    st.session_state.show_latest = not st.session_state.show_latest
    st.session_state.show_history = False
    st.session_state.show_chart = False
    st.session_state.show_predict = False
    st.session_state.show_buy = False
    st.session_state.show_sell = False
    
if col2.button("Show Historical Data", use_container_width=True):
    st.session_state.show_history = not st.session_state.show_history
    st.session_state.show_latest = False
    st.session_state.show_chart = False
    st.session_state.show_predict = False
    st.session_state.show_buy = False
    st.session_state.show_sell = False
    
if col3.button("Show Chart", use_container_width=True):
    st.session_state.show_chart = not st.session_state.show_chart
    st.session_state.show_latest = False
    st.session_state.show_history = False
    st.session_state.show_predict = False
    st.session_state.show_buy = False
    st.session_state.show_sell = False

# Button - Row 2
col4, col5, col6 = st.columns(3)
if col4.button("Predict Future Prices", use_container_width=True, type="primary"):
    st.session_state.show_predict = not st.session_state.show_predict
    st.session_state.show_latest = False
    st.session_state.show_history = False
    st.session_state.show_chart = False
    st.session_state.show_buy = False
    st.session_state.show_sell = False
    
if col5.button("Buy Recommendation", use_container_width=True):
    st.session_state.show_buy = not st.session_state.show_buy
    st.session_state.show_latest = False
    st.session_state.show_history = False
    st.session_state.show_chart = False
    st.session_state.show_predict = False
    st.session_state.show_sell = False
    
if col6.button("Sell Analysis", use_container_width=True):
    st.session_state.show_sell = not st.session_state.show_sell
    st.session_state.show_latest = False
    st.session_state.show_history = False
    st.session_state.show_chart = False
    st.session_state.show_predict = False
    st.session_state.show_buy = False

def is_pse_stock(symbol):
    """Check if this is a Philippine stock"""
    return symbol.upper().endswith('.PS')

@st.cache_data(ttl=3600)
def fetch_yahoo_data(symbol, period="1y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return None, None, False
        
        info = stock.info
        
        # Check if it's a Philippine stock
        is_pse = symbol.upper().endswith('.PS')
        
        return hist, info, is_pse
    except Exception as e:
        st.error(f"Error fetching Yahoo data: {e}")
        return None, None, False

def fetch_data_hybrid(symbol):
    """Smart fetch: automatically choose the right API"""
    # Add .PS if it looks like a PSE stock without suffix
    pse_stocks = ['JFC', 'SM', 'BDO', 'ALI', 'MBT', 'BPI', 'SMPH', 'MEG', 'TEL', 'GLO',
                  'AC', 'AGI', 'MER', 'DMC', 'URC', 'PGOLD', 'LTG', 'ICT', 'RLC', 'AEV',
                  'COL', 'BLOOM', 'CNPF', 'CEB', 'SMDC', 'FGEN', 'AP', 'EMI', 'HOUSE', 'CLI']
    
    if symbol.upper() in pse_stocks and not symbol.upper().endswith('.PS'):
        symbol = symbol.upper() + '.PS'
        st.info(f"🇵🇭 Auto-corrected to PSE format: {symbol}")
    
    if symbol.upper().endswith('.PS'):
        st.info(f"🇵🇭 Fetching Philippine stock: {symbol} - Using Yahoo Finance PSE data")
    else:
        st.info(f"🌍 Fetching international stock: {symbol} - Using Yahoo Finance")
    
    return fetch_yahoo_data(symbol)

if symbol:
    with st.spinner(f"Fetching data for {symbol.upper()}..."):
        hist_df, stock_info, is_pse = fetch_data_hybrid(symbol)

    if hist_df is not None and stock_info is not None:
        # Standardize column names
        hist_df = hist_df.reset_index()
        hist_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        hist_df.set_index('Date', inplace=True)
        hist_df = hist_df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Check if stock is in PHP (PSE stocks) or needs conversion
        if is_pse:
            # PSE stocks from Yahoo are already in PHP
            hist_df['Open_PHP'] = hist_df['Open']
            hist_df['High_PHP'] = hist_df['High']
            hist_df['Low_PHP'] = hist_df['Low']
            hist_df['Close_PHP'] = hist_df['Close']
        else:
            # Convert USD to PHP for international stocks
            hist_df['Open_PHP'] = hist_df['Open'] * USD_TO_PHP
            hist_df['High_PHP'] = hist_df['High'] * USD_TO_PHP
            hist_df['Low_PHP'] = hist_df['Low'] * USD_TO_PHP
            hist_df['Close_PHP'] = hist_df['Close'] * USD_TO_PHP

        hist_df = hist_df.sort_index(ascending=False)

        # No warning needed - Yahoo Finance has full historical data for PSE

        # LATEST PRICE
        if st.session_state.show_latest:
            st.subheader("Latest Stock Price")

            latest_date = hist_df.index[0]
            latest_row = hist_df.loc[latest_date]

            if len(hist_df) > 1:
                prev_close = hist_df.iloc[1]["Close_PHP"]
                change = latest_row["Close_PHP"] - prev_close
                change_pct = (change / prev_close) * 100
                change_color = "green" if change >= 0 else "red"
                change_symbol = "▲" if change >= 0 else "▼"
            else:
                change = 0
                change_pct = 0
                change_color = "gray"
                change_symbol = "●"

            company_name = stock_info.get('longName', symbol.upper())

            st.markdown(
                f"""
                <div style="padding: 20px; background-color: #f7f7f7; border-radius: 10px; border: 1px solid #ddd;">
                    <h3 style="margin-top:0;">Latest data for <b>{company_name}</b> ({symbol.upper()})</h3>
                    <p style="font-size: 14px; color: #666;">Date: {latest_date.strftime('%Y-%m-%d %H:%M')}</p>
                    <p style="font-size: 24px; font-weight: bold; color: {change_color};">
                        ₱{latest_row["Close_PHP"]:.2f} 
                        <span style="font-size: 18px;">{change_symbol} {change:.2f} ({change_pct:+.2f}%)</span>
                    </p>
                    <p style="font-size: 12px; color: #999;">{'Already in PHP' if is_pse else f'Exchange Rate: 1 USD = ₱{USD_TO_PHP:.2f}'}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if is_pse:
                latest_df = pd.DataFrame({
                    "Metric": ["Open", "High", "Low", "Close", "Volume"],
                    "Value (PHP)": [
                        f"₱{latest_row['Open_PHP']:.2f}",
                        f"₱{latest_row['High_PHP']:.2f}",
                        f"₱{latest_row['Low_PHP']:.2f}",
                        f"₱{latest_row['Close_PHP']:.2f}",
                        f"{int(latest_row['Volume']):,}"
                    ]
                })
            else:
                latest_df = pd.DataFrame({
                    "Metric": ["Open", "High", "Low", "Close", "Volume"],
                    "Value (PHP)": [
                        f"₱{latest_row['Open_PHP']:.2f}",
                        f"₱{latest_row['High_PHP']:.2f}",
                        f"₱{latest_row['Low_PHP']:.2f}",
                        f"₱{latest_row['Close_PHP']:.2f}",
                        f"{int(latest_row['Volume']):,}"
                    ],
                    "Value (USD)": [
                        f"${latest_row['Open']:.2f}",
                        f"${latest_row['High']:.2f}",
                        f"${latest_row['Low']:.2f}",
                        f"${latest_row['Close']:.2f}",
                        f"{int(latest_row['Volume']):,}"
                    ]
                })

            st.table(latest_df)
            
            with st.expander("Company Information"):
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
                    st.write(f"**Market Cap:** {stock_info.get('marketCap', 'N/A')}")
                with col_info2:
                    st.write(f"**52 Week High:** {stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
                    st.write(f"**52 Week Low:** {stock_info.get('fiftyTwoWeekLow', 'N/A')}")
                    st.write(f"**Website:** {stock_info.get('website', 'N/A')}")

        # HISTORICAL DATA
        if st.session_state.show_history:
            st.subheader("Historical Stock Prices")
            
            if len(hist_df) >= 30:
                days = st.selectbox("Select time period:", [30, 60, 90, 180, 365, "All"], index=1)
                
                if days == "All":
                    display_df = hist_df
                else:
                    display_df = hist_df.head(days)
            else:
                display_df = hist_df
                st.info(f"Showing all available data ({len(hist_df)} days)")
            
            display_columns = ['Open_PHP', 'High_PHP', 'Low_PHP', 'Close_PHP', 'Volume']
            display_df_show = display_df[display_columns].copy()
            display_df_show.columns = ['Open (PHP)', 'High (PHP)', 'Low (PHP)', 'Close (PHP)', 'Volume']
            
            st.dataframe(display_df_show.style.format({
                "Open (PHP)": "₱{:.2f}",
                "High (PHP)": "₱{:.2f}",
                "Low (PHP)": "₱{:.2f}",
                "Close (PHP)": "₱{:.2f}",
                "Volume": "{:,.0f}"
            }), use_container_width=True)
            
            csv = display_df_show.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{symbol}_stock_data.csv",
                mime="text/csv"
            )

        # CHARTS
        if st.session_state.show_chart:
            st.subheader("Stock Charts")

            if len(hist_df) >= 30:
                days = st.selectbox("Chart time period:", [30, 60, 90, 180, 365, "All"], index=1, key="chart_days")
                
                if days == "All":
                    chart_df = hist_df.sort_index(ascending=True)
                else:
                    chart_df = hist_df.head(days).sort_index(ascending=True)
            else:
                chart_df = hist_df.sort_index(ascending=True)

            st.write("### Closing Price Trend")
            st.line_chart(chart_df["Close_PHP"])

            st.write("### Candlestick Chart")
            candle_fig = go.Figure(data=[go.Candlestick(
                x=chart_df.index,
                open=chart_df["Open_PHP"],
                high=chart_df["High_PHP"],
                low=chart_df["Low_PHP"],
                close=chart_df["Close_PHP"],
                name=symbol.upper()
            )])
            candle_fig.update_layout(
                height=600,
                xaxis_title="Date",
                yaxis_title="Price (PHP)",
                hovermode="x unified"
            )
            st.plotly_chart(candle_fig, use_container_width=True)

            st.write("### Volume Chart")
            st.bar_chart(chart_df["Volume"])

        # PREDICT
        if st.session_state.show_predict:
            if len(hist_df) < 30:
                st.error("❌ Insufficient historical data for prediction. Need at least 30 days of data.")
            else:
                st.subheader("Stock Price Prediction (Linear Regression)")
                
                st.warning("**Disclaimer:** This is a simple linear regression model for educational purposes only.")
                
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    max_training = min(len(hist_df), 365)
                    training_days = st.slider("Training period (days):", 30, max_training, min(90, max_training))
                with col_pred2:
                    predict_days = st.slider("Predict ahead (days):", 1, 90, 30)
                
                pred_df = hist_df.sort_index(ascending=True).tail(training_days).copy()
                pred_df['Day_Num'] = np.arange(len(pred_df))
                
                X = pred_df['Day_Num'].values.reshape(-1, 1)
                y = pred_df['Close_PHP'].values
                
                model = LinearRegression()
                model.fit(X, y)
                pred_df['Predicted'] = model.predict(X)
                
                future_days = np.arange(len(pred_df), len(pred_df) + predict_days).reshape(-1, 1)
                future_prices = model.predict(future_days)
                
                last_date = pred_df.index[-1]
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=predict_days, freq='D')
                
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Price_PHP': future_prices
                })
                future_df.set_index('Date', inplace=True)
                
                from sklearn.metrics import r2_score, mean_absolute_error
                r2 = r2_score(y, pred_df['Predicted'])
                mae = mean_absolute_error(y, pred_df['Predicted'])
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("R² Score", f"{r2:.4f}")
                with col_m2:
                    st.metric("Mean Abs Error", f"₱{mae:.2f}")
                with col_m3:
                    st.metric("Current Price", f"₱{pred_df['Close_PHP'].iloc[-1]:.2f}")
                with col_m4:
                    predicted_price = future_prices[-1]
                    price_change = predicted_price - pred_df['Close_PHP'].iloc[-1]
                    st.metric(f"Predicted ({predict_days}d)", f"₱{predicted_price:.2f}", f"{price_change:+.2f}")
                
                if r2 > 0.7:
                    st.success(f"Strong linear trend detected (R² = {r2:.4f})")
                elif r2 > 0.4:
                    st.info(f"Moderate linear trend (R² = {r2:.4f})")
                else:
                    st.warning(f"Weak linear trend (R² = {r2:.4f})")
                
                st.write("### Future Price Predictions Table")
                
                future_display = future_df.copy()
                future_display['Predicted_Price_PHP'] = future_display['Predicted_Price_PHP'].apply(lambda x: f"₱{x:.2f}")
                if not is_pse:
                    future_display['Predicted_Price_USD'] = (future_df['Predicted_Price_PHP'] / USD_TO_PHP).apply(lambda x: f"${x:.2f}")
                future_display.index = future_display.index.strftime('%Y-%m-%d')
                future_display.index.name = 'Date'
                
                st.dataframe(future_display, use_container_width=True, height=400)
                
                csv_pred = future_df.to_csv()
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_pred,
                    file_name=f"{symbol}_predictions.csv",
                    mime="text/csv"
                )
                
                st.write("---")
                st.write("### Price Prediction Chart")
                
                fig_pred = go.Figure()
                
                fig_pred.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Close_PHP'],
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='blue', width=2)
                ))
                
                fig_pred.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Predicted'],
                    mode='lines',
                    name='Fitted Line',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                
                fig_pred.add_trace(go.Scatter(
                    x=future_df.index,
                    y=future_df['Predicted_Price_PHP'],
                    mode='lines+markers',
                    name='Future Prediction',
                    line=dict(color='red', width=2, dash='dot'),
                    marker=dict(size=6)
                ))
                
                fig_pred.update_layout(
                    height=600,
                    xaxis_title="Date",
                    yaxis_title="Price (PHP)",
                    hovermode="x unified",
                    showlegend=True
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                st.write("### Model Details")
                
                slope = model.coef_[0]
                intercept = model.intercept_
                st.code(f"Price = ₱{intercept:.2f} + ₱{slope:.2f} × Day")
                
                trend_direction = "upward" if slope > 0 else "downward"
                st.write(f"**Trend:** {trend_direction} trend of **₱{abs(slope):.2f} per day**.")

        # BUY RECOMMENDATION
        if st.session_state.show_buy:
            if len(hist_df) < 30:
                st.error("❌ Insufficient historical data for recommendation. Need at least 30 days.")
            else:
                st.subheader("Buy Recommendation Analysis")
                
                st.info("This analysis uses linear regression prediction.")
                
                col_buy1, col_buy2 = st.columns(2)
                with col_buy1:
                    num_shares = st.number_input("How many shares do you plan to buy?", min_value=1, value=100, step=1)
                with col_buy2:
                    investment_horizon = st.slider("Investment time horizon (days):", 7, 90, 30)
                
                current_price_php = hist_df.iloc[0]['Close_PHP']
                
                pred_df = hist_df.sort_index(ascending=True).tail(min(90, len(hist_df))).copy()
                pred_df['Day_Num'] = np.arange(len(pred_df))
                
                X = pred_df['Day_Num'].values.reshape(-1, 1)
                y = pred_df['Close_PHP'].values
                model = LinearRegression()
                model.fit(X, y)
                
                future_day = np.array([[len(pred_df) + investment_horizon]])
                predicted_price_php = model.predict(future_day)[0]
                
                price_change = predicted_price_php - current_price_php
                price_change_pct = (price_change / current_price_php) * 100
                
                total_investment = num_shares * current_price_php
                potential_value = num_shares * predicted_price_php
                potential_profit = potential_value - total_investment
                
                pred_df['Predicted'] = model.predict(X)
                from sklearn.metrics import r2_score
                r2 = r2_score(y, pred_df['Predicted'])
                
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.metric("Current Price", f"₱{current_price_php:.2f}")
                with col_p2:
                    st.metric(f"Predicted ({investment_horizon}d)", f"₱{predicted_price_php:.2f}", f"{price_change_pct:+.2f}%")
                with col_p3:
                    st.metric("R² Score", f"{r2:.4f}")
                
                st.write("---")
                st.write("### Investment Analysis")
                
                investment_df = pd.DataFrame({
                    "Item": [
                        "Number of Shares",
                        "Current Price per Share",
                        "Total Investment",
                        f"Predicted Price ({investment_horizon}d)",
                        "Predicted Value",
                        "Potential Profit/Loss",
                        "Return (%)"
                    ],
                    "Amount": [
                        f"{num_shares:,}",
                        f"₱{current_price_php:.2f}",
                        f"₱{total_investment:,.2f}",
                        f"₱{predicted_price_php:.2f}",
                        f"₱{potential_value:,.2f}",
                        f"₱{potential_profit:+,.2f}",
                        f"{price_change_pct:+.2f}%"
                    ]
                })
                
                st.table(investment_df)
                
                st.write("### Recommendation")
                
                is_uptrend = price_change > 0
                is_reliable = r2 > 0.5
                is_significant = price_change_pct > 5
                
                if is_uptrend and is_reliable and is_significant:
                    st.success("✅ STRONG BUY")
                    st.write(f"- Predicted {price_change_pct:.2f}% increase")
                    st.write(f"- Good reliability (R² = {r2:.4f})")
                elif is_uptrend and is_reliable:
                    st.info("ℹ️ MODERATE BUY")
                    st.write(f"- Predicted {price_change_pct:.2f}% increase")
                elif is_uptrend:
                    st.warning("⚠️ CAUTIOUS - LOW CONFIDENCE")
                    st.write(f"- Low reliability (R² = {r2:.4f})")
                else:
                    st.error("❌ NOT RECOMMENDED")
                    st.write(f"- Predicted {price_change_pct:.2f}% decrease")

        # SELL ANALYSIS
        if st.session_state.show_sell:
            if len(hist_df) < 30:
                st.error("❌ Insufficient historical data for analysis. Need at least 30 days.")
            else:
                st.subheader("Sell Analysis & Profit Calculator")
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    shares_owned = st.number_input("Shares owned:", min_value=1, value=100, step=1)
                with col_s2:
                    purchase_price = st.number_input("Purchase price (PHP):", min_value=0.01, value=1000.0, step=0.01)
                with col_s3:
                    hold_days = st.slider("Hold before selling (days):", 7, 90, 30)
                
                current_price_php = hist_df.iloc[0]['Close_PHP']
                
                total_purchase = shares_owned * purchase_price
                total_current = shares_owned * current_price_php
                current_profit = total_current - total_purchase
                current_profit_pct = (current_profit / total_purchase) * 100
                
                pred_df = hist_df.sort_index(ascending=True).tail(min(90, len(hist_df))).copy()
                pred_df['Day_Num'] = np.arange(len(pred_df))
                
                X = pred_df['Day_Num'].values.reshape(-1, 1)
                y = pred_df['Close_PHP'].values
                model = LinearRegression()
                model.fit(X, y)
                
                future_day = np.array([[len(pred_df) + hold_days]])
                predicted_price = model.predict(future_day)[0]
                
                total_predicted = shares_owned * predicted_price
                predicted_profit = total_predicted - total_purchase
                predicted_profit_pct = (predicted_profit / total_purchase) * 100
                
                additional = predicted_profit - current_profit
                
                st.write("### Current Position")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Purchase", f"₱{purchase_price:.2f}")
                with col_m2:
                    st.metric("Current", f"₱{current_price_php:.2f}")
                with col_m3:
                    st.metric("Profit/Loss", f"₱{current_profit:,.2f}", f"{current_profit_pct:+.2f}%")
                with col_m4:
                    st.metric(f"Predicted ({hold_days}d)", f"₱{predicted_price:.2f}")
                
                st.write("---")
                st.write("### Analysis")
                
                analysis_df = pd.DataFrame({
                    "Scenario": ["Sell Now", f"Sell in {hold_days}d"],
                    "Price": [f"₱{current_price_php:.2f}", f"₱{predicted_price:.2f}"],
                    "Total Value": [f"₱{total_current:,.2f}", f"₱{total_predicted:,.2f}"],
                    "Profit": [f"₱{current_profit:+,.2f}", f"₱{predicted_profit:+,.2f}"],
                    "Return": [f"{current_profit_pct:+.2f}%", f"{predicted_profit_pct:+.2f}%"]
                })
                
                st.table(analysis_df)
                
                if additional > 0:
                    st.info(f"💡 Potential additional gain: ₱{additional:,.2f}")
                else:
                    st.warning(f"⚠️ Potential additional loss: ₱{abs(additional):,.2f}")
                
                st.write("### Recommendation")
                
                has_profit = current_profit > 0
                will_increase = predicted_price > current_price_php
                
                if has_profit and current_profit_pct > 10 and not will_increase:
                    st.success("✅ STRONG SELL - Lock in profits")
                elif has_profit and will_increase:
                    st.info("ℹ️ HOLD - More gains expected")
                elif not has_profit and will_increase:
                    st.warning("⚠️ HOLD - Wait for recovery")
                else:
                    st.error("❌ MINIMIZE LOSS")

    else:
        st.error(f"Unable to fetch data for: {symbol.upper()}")
        st.info("**Tips:**\n- For international stocks: Use symbols like AAPL, TSLA, GOOGL\n- For Philippine stocks: Add .PS suffix like JFC.PS, SM.PS, BDO.PS, or just type JFC, SM, BDO (auto-adds .PS)")
else:
    st.write("Enter a stock symbol above to get started.")

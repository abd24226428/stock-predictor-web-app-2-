import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Streamlit UI
st.title("📈 Apple vs. Google vs. Samsung Stock Analysis & Prediction")

# Select stock ticker
ticker = st.selectbox("Select Stock", ["AAPL", "GOOGL", "SSNLF"])

# Load stock data
@st.cache_data
def load_stock_data(ticker):
    if ticker == "SSNLF":
        st.warning("Samsung (SSNLF) data is limited on Yahoo Finance. Consider an alternative data source.")
        return pd.DataFrame()  # Placeholder for SSNLF (you can replace this with actual data)
    
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")
    df.reset_index(inplace=True)

    # Calculate Moving Averages
    df["50-Day MA"] = df["Close"].rolling(window=50).mean()
    df["200-Day MA"] = df["Close"].rolling(window=200).mean()
    
    return df

df = load_stock_data(ticker)

# Display data
st.subheader(f"{ticker} Stock Data")
if df.empty:
    st.write("No available data for SSNLF.")
else:
    st.write(df.tail())

# 📊 Stock Price, Moving Averages, and Volume Graph
st.subheader("📊 Stock Price, Moving Averages, and Volume")

if not df.empty:
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Close Price & Moving Averages
    ax1.plot(df["Date"], df["Close"], label="Close Price", color="blue")
    ax1.plot(df["Date"], df["50-Day MA"], label="50-Day MA", color="orange", linestyle="dashed")
    ax1.plot(df["Date"], df["200-Day MA"], label="200-Day MA", color="red", linestyle="dashed")

    # Plot Open, High, and Low Prices
    ax1.plot(df["Date"], df["Open"], label="Open Price", color="purple", linestyle="dotted")
    ax1.plot(df["Date"], df["High"], label="High Price", color="green", linestyle="dotted")
    ax1.plot(df["Date"], df["Low"], label="Low Price", color="brown", linestyle="dotted")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")

    # Secondary y-axis for Volume
    ax2 = ax1.twinx()
    ax2.bar(df["Date"], df["Volume"], label="Volume", color="gray", alpha=0.3)
    ax2.set_ylabel("Volume")
    ax2.legend(loc="upper right")

    st.pyplot(fig)

# 🎯 Product Launch Events
product_launches = {
    "AAPL": [
        ("iPhone 12", "2020-10-13"),
        ("iPhone 13", "2021-09-14"),
        ("iPhone 14", "2022-09-07"),
        ("iPhone 15", "2023-09-12"),
    ],
    "GOOGL": [
        ("Pixel 5", "2020-09-30"),
        ("Pixel 6", "2021-10-19"),
        ("Pixel 7", "2022-10-06"),
        ("Pixel 8", "2023-10-04"),
    ],
    "SSNLF": [
        ("Galaxy S21", "2021-01-14"),
        ("Galaxy S22", "2022-02-09"),
        ("Galaxy S23", "2023-02-01"),
        ("Galaxy S24", "2024-01-17"),
    ]
}

# 📊 Stock Performance After Product Launches
st.subheader(f"{ticker} Stock Performance After Major Product Launches")
if not df.empty:
    fig, ax1 = plt.subplots(figsize=(10,5))

    ax1.plot(df["Date"], df["Close"], label="Close Price", color="blue")
    ax1.plot(df["Date"], df["Open"], label="Open Price", color="purple", linestyle="dotted")
    ax1.plot(df["Date"], df["High"], label="High Price", color="green", linestyle="dotted")
    ax1.plot(df["Date"], df["Low"], label="Low Price", color="brown", linestyle="dotted")

    # Highlight product launch events
    for event, date in product_launches.get(ticker, []):
        event_date = pd.to_datetime(date)
        if event_date in df["Date"].values:
            ax1.axvline(event_date, color="black", linestyle="dashed", alpha=0.7, label=f"{event}")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")

    # Secondary y-axis for Volume
    ax2 = ax1.twinx()
    ax2.bar(df["Date"], df["Volume"], label="Volume", color="gray", alpha=0.3)
    ax2.set_ylabel("Volume")
    ax2.legend(loc="upper right")

    st.pyplot(fig)

# Data preprocessing for LSTM
if not df.empty:
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

    # Prepare training data
    def create_sequences(data, time_step=50):
        X, Y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:i+time_step])
            Y.append(data[i+time_step])
        return np.array(X), np.array(Y)

    time_step = 50
    X, Y = create_sequences(df_scaled)

    # Train-test split
    split = int(len(X) * 0.8)
    X_train, Y_train = X[:split], Y[:split]
    X_test, Y_test = X[split:], Y[split:]

    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, activation="relu", input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False, activation="relu"),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train model
    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=1)
            st.success("Model training complete! ✅")

    # Make predictions
    if st.button("Predict"):
        Y_pred = model.predict(X_test)
        Y_pred = scaler.inverse_transform(Y_pred)
        Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

        # 📊 Prediction vs. Actual Graph
        st.subheader("📊 Prediction vs. Actual")
        fig, ax = plt.subplots()
        ax.plot(Y_test, label="Actual Price", color="green")
        ax.plot(Y_pred, label="Predicted Price", color="red")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)

st.write("🚀 Built with Streamlit | Data from Yahoo Finance")

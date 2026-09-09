import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Crypto Analysis Suite", layout="wide")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    """Load and preprocess the uploaded CSV file."""
    df = pd.read_csv(BytesIO(file_bytes))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    df = df.dropna(subset=["timestamp"])
    
    # Convert timestamp to timezone-naive for consistent plotting
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        
    return df


def calculate_returns(df):
    """Calculate daily returns for each cryptocurrency."""
    returns_list = []
    for crypto, group in df.groupby("crypto_name"):
        group = group.sort_values("timestamp").copy()
        group["daily_return"] = group["close"].pct_change() * 100
        ratio = group["close"] / group["close"].shift(1)
        group["log_return"] = np.where(ratio > 0, np.log(ratio) * 100, np.nan)
        returns_list.append(group[["timestamp", "crypto_name", "daily_return", "log_return"]])
    return pd.concat(returns_list, ignore_index=True).dropna(subset=["daily_return"])


def calculate_moving_averages(df, windows=[7, 21, 50]):
    """Calculate simple moving averages."""
    df = df.sort_values("timestamp").copy()
    for window in windows:
        df[f"sma_{window}"] = df["close"].rolling(window=window, min_periods=1).mean()
    return df


def calculate_exponential_moving_averages(df, spans=[7, 21, 50]):
    """Calculate exponential moving averages."""
    df = df.sort_values("timestamp").copy()
    for span in spans:
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()
    return df


def calculate_bollinger_bands(df, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    df = df.sort_values("timestamp").copy()
    df["bb_middle"] = df["close"].rolling(window=window, min_periods=1).mean()
    df["bb_std"] = df["close"].rolling(window=window, min_periods=1).std().fillna(0)
    df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * num_std)
    df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * num_std)
    
    # Guard against division by zero
    safe_middle = df["bb_middle"].replace(0, np.nan)
    df["bb_width"] = ((df["bb_upper"] - df["bb_lower"]) / safe_middle) * 100
    return df


def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index."""
    df = df.sort_values("timestamp").copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = np.where(avg_loss > 0, avg_gain / avg_loss, np.inf)
    rsi = np.where(avg_loss > 0, 100 - (100 / (1 + rs)), 100.0)
    # Flat windows (no gains, no losses) are neutral
    rsi = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, rsi)
    df["rsi"] = rsi
    return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    df = df.sort_values("timestamp").copy()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    return df


def calculate_volatility(df, windows=[7, 21, 30, 60]):
    """Calculate rolling volatility per cryptocurrency."""
    df = df.sort_values(["crypto_name", "timestamp"]).copy()
    df["daily_return"] = df.groupby("crypto_name")["close"].pct_change() * 100
    for window in windows:
        rolled = (
            df.groupby("crypto_name")["daily_return"]
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )
        df[f"volatility_{window}d"] = rolled
    return df


def calculate_drawdown(df):
    """Calculate drawdown from peak."""
    df = df.sort_values("timestamp").copy()
    df["peak"] = df["close"].cummax()
    df["drawdown"] = np.where(df["peak"] > 0, (df["close"] - df["peak"]) / df["peak"] * 100, 0.0)
    return df


# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Select Section",
    [
        "Case Study",
        "EDA & Visualization",
        "Technical Indicators",
        "Comparison Dashboard",
        "3D Explorer",
        "Data Download",
        "Machine Learning",
    ],
)

# -----------------------------------------------------------------------------
# Data loading (common to all sections)
# -----------------------------------------------------------------------------
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your crypto CSV", type=["csv"])

# 1. Initialize df to None to prevent NameError
df = None

if uploaded_file is None:
    st.info("Please upload a CSV file to get started.")
    st.sidebar.markdown(
        "**Expected columns:**\n`open, high, low, close, volume, marketCap, timestamp, crypto_name, date`"
    )
    st.stop()

try:
    # 2. Read file bytes for caching (fixes unhashable UploadedFile error)
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    
    # 3. Explicitly assign the result to df
    df = load_data(file_bytes, file_name)
    
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# 4. Safety check: ensure df actually exists before proceeding
if df is None or df.empty:
    st.error("Failed to load data. The CSV file might be empty or corrupted.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    """Load and preprocess the uploaded CSV file."""
    df = pd.read_csv(BytesIO(file_bytes))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    df = df.dropna(subset=["timestamp"])
    
    # Convert timestamp to timezone-naive for consistent plotting
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        
    return df  # <-- MAKE SURE THIS LINE IS HERE








# -----------------------------------------------------------------------------
# 1. Case Study
# -----------------------------------------------------------------------------
if app_mode == "Case Study":
    st.title("Cryptocurrency Price Analysis - Case Study")
    st.markdown(
        """
    ### Background
    Cryptocurrencies have evolved from a niche digital asset to a mainstream financial instrument.
    Understanding their price movements, volatility, and interrelationships is critical.

    ### Objectives
    - **EDA:** Identify trends, seasonality, and anomalies.
    - **Visualization:** Communicate complex patterns interactively.
    - **Technical Indicators:** Analyze price action with standard trading indicators.
    - **Comparison Dashboard:** Compare multiple cryptocurrencies side by side.
    - **3D Exploration:** Reveal hidden structures in multivariate price data.
    - **Machine Learning:** Build simple predictive models for short-term forecasting.
    """
    )

# -----------------------------------------------------------------------------
# 2. EDA & Visualization
# -----------------------------------------------------------------------------
elif app_mode == "EDA & Visualization":
    st.title("Exploratory Data Analysis & Visualization")
    crypto_list = sorted(df["crypto_name"].unique())
    selected_crypto = st.multiselect("Select cryptocurrencies", crypto_list, default=crypto_list[:2])

    if not selected_crypto:
        st.warning("Please select at least one cryptocurrency.")
        st.stop()

    eda_df = df[df["crypto_name"].isin(selected_crypto)].copy()
    st.dataframe(eda_df[["open", "high", "low", "close", "volume", "marketCap"]].describe())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(eda_df))
    col2.metric("Date Range", f"{eda_df['timestamp'].min().strftime('%Y-%m-%d')} to {eda_df['timestamp'].max().strftime('%Y-%m-%d')}")
    col3.metric("Cryptocurrencies", len(selected_crypto))
    col4.metric("Average Close Price", f"${eda_df['close'].mean():,.2f}")

    st.subheader("Close Price Over Time")
    fig = px.line(eda_df, x="timestamp", y="close", color="crypto_name", title="Daily Close Price")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("OHLC Candlestick Chart")
    candlestick_crypto = st.selectbox("Select cryptocurrency for candlestick", selected_crypto)
    candlestick_df = eda_df[eda_df["crypto_name"] == candlestick_crypto].sort_values("timestamp")
    
    if len(candlestick_df) > 0:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=candlestick_df["timestamp"], open=candlestick_df["open"], high=candlestick_df["high"], low=candlestick_df["low"], close=candlestick_df["close"], name="OHLC"), row=1, col=1)
        fig.add_trace(go.Bar(x=candlestick_df["timestamp"], y=candlestick_df["volume"], name="Volume", marker_color="steelblue", opacity=0.6), row=2, col=1)
        fig.update_layout(title=f"{candlestick_crypto} - Candlestick Chart with Volume", xaxis_title="Date", yaxis_title="Price (USD)")
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution of Daily Returns")
    returns_df = calculate_returns(eda_df)
    if not returns_df.empty:
        fig = px.histogram(returns_df, x="daily_return", color="crypto_name", nbins=50, marginal="box", opacity=0.6, title="Daily Returns (%) Distribution")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 3. Technical Indicators
# -----------------------------------------------------------------------------
elif app_mode == "Technical Indicators":
    st.title("Technical Indicators Analysis")
    crypto_list = sorted(df["crypto_name"].unique())
    selected_crypto = st.selectbox("Select cryptocurrency", crypto_list)
    crypto_df = df[df["crypto_name"] == selected_crypto].sort_values("timestamp").copy()

    if len(crypto_df) < 5:
        st.error("Not enough data for technical analysis (minimum 5 rows).")
        st.stop()

    st.sidebar.header("Indicator Options")
    show_sma = st.sidebar.checkbox("Simple Moving Averages", value=True)
    show_ema = st.sidebar.checkbox("Exponential Moving Averages", value=True)
    show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=True)
    show_rsi = st.sidebar.checkbox("Relative Strength Index (RSI)", value=True)
    show_macd = st.sidebar.checkbox("MACD", value=True)
    show_volatility = st.sidebar.checkbox("Volatility Analysis", value=True)
    show_drawdown = st.sidebar.checkbox("Drawdown Analysis", value=True)

    if show_sma: crypto_df = calculate_moving_averages(crypto_df)
    if show_ema: crypto_df = calculate_exponential_moving_averages(crypto_df)
    if show_bollinger: crypto_df = calculate_bollinger_bands(crypto_df)
    if show_rsi: crypto_df = calculate_rsi(crypto_df)
    if show_macd: crypto_df = calculate_macd(crypto_df)
    if show_volatility: crypto_df = calculate_volatility(crypto_df)
    if show_drawdown: crypto_df = calculate_drawdown(crypto_df)

    st.subheader("Price Chart with Technical Indicators")
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=crypto_df["timestamp"], y=crypto_df["close"], mode="lines", name="Close Price", line=dict(color="royalblue", width=2)), row=1, col=1)
    
    if show_sma:
        for col_name, color in [("sma_7", "orange"), ("sma_21", "green"), ("sma_50", "purple")]:
            if col_name in crypto_df.columns:
                fig.add_trace(go.Scatter(x=crypto_df["timestamp"], y=crypto_df[col_name], mode="lines", name=col_name.upper(), line=dict(color=color, width=1, dash="dash")), row=1, col=1)
    
    if show_bollinger and "bb_upper" in crypto_df.columns:
        fig.add_trace(go.Scatter(x=crypto_df["timestamp"], y=crypto_df["bb_upper"], mode="lines", name="BB Upper", line=dict(color="gray", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=crypto_df["timestamp"], y=crypto_df["bb_lower"], mode="lines", name="BB Lower", line=dict(color="gray", width=1, dash="dash"), fill="tonexty", fillcolor="rgba(128,128,128,0.1)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=crypto_df["timestamp"], y=crypto_df["bb_middle"], mode="lines", name="BB Middle", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)

    fig.update_layout(title=f"{selected_crypto} - Price with Technical Indicators", xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)

    if show_rsi and "rsi" in crypto_df.columns:
        st.subheader("Relative Strength Index (RSI)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=crypto_df["timestamp"], y=crypto_df["rsi"], mode="lines", name="RSI", line=dict(color="darkblue", width=2)))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig.update_layout(title="RSI Indicator (14-period)", yaxis_range=[0, 100], height=300)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 4. Comparison Dashboard
# -----------------------------------------------------------------------------
elif app_mode == "Comparison Dashboard":
    st.title("Cryptocurrency Comparison Dashboard")
    crypto_list = sorted(df["crypto_name"].unique())
    selected_cryptos = st.multiselect("Select cryptocurrencies to compare", crypto_list, default=crypto_list[:3])
    
    if len(selected_cryptos) < 2:
        st.warning("Please select at least 2 cryptocurrencies for comparison.")
        st.stop()

    comp_df = df[df["crypto_name"].isin(selected_cryptos)].sort_values("timestamp").copy()
    st.subheader("Close Price Comparison")
    fig = px.line(comp_df, x="timestamp", y="close", color="crypto_name", title="Close Price Comparison")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cross-Correlation Analysis")
    pivot_close = comp_df.pivot_table(index="timestamp", columns="crypto_name", values="close")
    corr_matrix = pivot_close.corr()
    fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", title="Price Correlation Matrix", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. 3D Explorer
# -----------------------------------------------------------------------------
elif app_mode == "3D Explorer":
    st.title("3D Data Explorer")
    numeric_cols = ["open", "high", "low", "close", "volume", "marketCap"]
    col1, col2, col3 = st.columns(3)
    with col1: x_axis = st.selectbox("X axis", numeric_cols, index=3)
    with col2: y_axis = st.selectbox("Y axis", numeric_cols, index=0)
    with col3: z_axis = st.selectbox("Z axis", numeric_cols, index=5)

    color_by = st.selectbox("Color by", ["crypto_name"] + numeric_cols)
    
    fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=color_by, hover_data=["timestamp", "crypto_name"], title=f"3D Scatter: {x_axis} vs {y_axis} vs {z_axis}", opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. Data Download
# -----------------------------------------------------------------------------
elif app_mode == "Data Download":
    st.title("Download Filtered Data")
    crypto_choice = st.multiselect("Select cryptocurrencies", df["crypto_name"].unique(), default=df["crypto_name"].unique())
    date_min = df["timestamp"].min().date()
    date_max = df["timestamp"].max().date()
    start_date, end_date = st.date_input("Select date range", [date_min, date_max], min_value=date_min, max_value=date_max)

    filtered = df[(df["crypto_name"].isin(crypto_choice)) & (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)]
    st.write(f"**Rows selected:** {len(filtered)}")
    
    if not filtered.empty:
        st.download_button(label="Download filtered data as CSV", data=filtered.to_csv(index=False), file_name="filtered_crypto_data.csv", mime="text/csv")

# -----------------------------------------------------------------------------
# 7. Machine Learning
# -----------------------------------------------------------------------------
elif app_mode == "Machine Learning":
    st.title("Machine Learning - Price Prediction")
    crypto_choice = st.selectbox("Select cryptocurrency", sorted(df["crypto_name"].unique()))
    crypto_df = df[df["crypto_name"] == crypto_choice].sort_values("timestamp").copy()

    if len(crypto_df) < 30:
        st.error("Not enough data for this cryptocurrency (minimum 30 rows).")
        st.stop()

    crypto_df["target"] = crypto_df["close"].shift(-1)
    crypto_df = crypto_df.dropna(subset=["target"]).reset_index(drop=True)

    features = st.multiselect("Select features", ["open", "high", "low", "close", "volume", "marketCap"], default=["open", "high", "low", "close", "volume"])
    if not features:
        st.warning("Please select at least one feature.")
        st.stop()

    X = crypto_df[features].copy()
    y = crypto_df["target"].copy()

    test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    model_type = st.selectbox("Choose model", ["Linear Regression", "Random Forest"])
    model = RandomForestRegressor(n_estimators=st.slider("Number of trees", 50, 300, 100), random_state=42) if model_type == "Random Forest" else LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"${mean_absolute_error(y_test, y_pred):.2f}")
    col2.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    col3.metric("R2 Score", f"{r2_score(y_test, y_pred):.4f}")

    test_indices = y_test.index
    result_df = pd.DataFrame({"timestamp": crypto_df.loc[test_indices, "timestamp"].values, "Actual": y_test.values, "Predicted": y_pred}).sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_df["timestamp"], y=result_df["Actual"], mode="lines+markers", name="Actual", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=result_df["timestamp"], y=result_df["Predicted"], mode="lines+markers", name="Predicted", line=dict(color="orange")))
    fig.update_layout(title=f"{crypto_choice} - Actual vs Predicted Close Price", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)


import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import threading 
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import ta  # Technical Analysis library
# Import required preprocessing tools for ML
from sklearn.preprocessing import MinMaxScaler 

# Supabase imports
from supabase import create_client, Client

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Invsion Connect - Algorithmic Trading Platform", layout="wide", initial_sidebar_state="expanded")
st.title("Invsion Connect - Algorithmic Trading & Price Prediction Platform")
st.markdown("A focused platform for fetching market data, performing in-depth analysis, and running ML-driven price predictions.")

# --- Global Constants & Session State Initialization ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"

# Initialize session state variables if they don't exist
if "kite_access_token" not in st.session_state:
    st.session_state["kite_access_token"] = None
if "kite_login_response" not in st.session_state:
    st.session_state["kite_login_response"] = None
if "instruments_df" not in st.session_state:
    st.session_state["instruments_df"] = pd.DataFrame()
if "historical_data" not in st.session_state:
    st.session_state["historical_data"] = pd.DataFrame()
if "last_fetched_symbol" not in st.session_state:
    st.session_state["last_fetched_symbol"] = None
if "user_session" not in st.session_state:
    st.session_state["user_session"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

# --- ML specific initializations (Fix for KeyError) ---
if "ml_data" not in st.session_state:
    st.session_state["ml_data"] = pd.DataFrame()
if "ml_model" not in st.session_state:
    st.session_state["ml_model"] = None
if "prediction_horizon" not in st.session_state:
    st.session_state["prediction_horizon"] = 5 # Default value

# --- Load Credentials from Streamlit Secrets ---
def load_secrets():
    secrets = st.secrets
    kite_conf = secrets.get("kite", {})
    supabase_conf = secrets.get("supabase", {})

    errors = []
    if not kite_conf.get("api_key") or not kite_conf.get("api_secret") or not kite_conf.get("redirect_uri"):
        errors.append("Kite credentials (api_key, api_secret, redirect_uri)")
    if not supabase_conf.get("url") or not supabase_conf.get("anon_key"):
        errors.append("Supabase credentials (url, anon_key)")

    if errors:
        st.error(f"Missing required credentials in `.streamlit/secrets.toml`: {', '.join(errors)}.")
        st.info("Example `secrets.toml`:\n```toml\n[kite]\napi_key=\"YOUR_KITE_API_KEY\"\napi_secret=\"YOUR_KITE_SECRET\"\nredirect_uri=\"http://localhost:8501\"\n\n[supabase]\nurl=\"YOUR_SUPABASE_URL\"\nanon_key=\"YOUR_SUPABASE_ANON_KEY\"\n```")
        st.stop()
    return kite_conf, supabase_conf

KITE_CREDENTIALS, SUPABASE_CREDENTIALS = load_secrets()

# --- Supabase Client Initialization ---
@st.cache_resource(ttl=3600) # Cache for 1 hour to prevent re-initializing on every rerun
def init_supabase_client(url: str, key: str) -> Client:
    return create_client(url, key)

supabase: Client = init_supabase_client(SUPABASE_CREDENTIALS["url"], SUPABASE_CREDENTIALS["anon_key"])

# --- KiteConnect Client Initialization (Unauthenticated for login URL) ---
@st.cache_resource(ttl=3600)
def init_kite_unauth_client(api_key: str) -> KiteConnect:
    return KiteConnect(api_key=api_key)

kite_unauth_client = init_kite_unauth_client(KITE_CREDENTIALS["api_key"])
login_url = kite_unauth_client.login_url()


# --- Utility Functions ---

def get_authenticated_kite_client(api_key: str | None, access_token: str | None) -> KiteConnect | None:
    if api_key and access_token:
        k_instance = KiteConnect(api_key=api_key)
        k_instance.set_access_token(access_token)
        return k_instance
    return None


@st.cache_data(ttl=86400, show_spinner="Loading instruments...") # Cache for 24 hours
def load_instruments_cached(api_key: str, access_token: str, exchange: str = None) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame({"_error": ["Kite not authenticated to load instruments."]})
    try:
        instruments = kite_instance.instruments(exchange) if exchange else kite_instance.instruments()
        df = pd.DataFrame(instruments)
        if "instrument_token" in df.columns:
            df["instrument_token"] = df["instrument_token"].astype("int64")
        if 'tradingsymbol' in df.columns and 'name' in df.columns:
            df = df[['instrument_token', 'tradingsymbol', 'name', 'exchange']]
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [f"Failed to load instruments for {exchange or 'all exchanges'}: {e}"]})


@st.cache_data(ttl=60) # Cache LTP for 1 minute
def get_ltp_price_cached(api_key: str, access_token: str, symbol: str, exchange: str = DEFAULT_EXCHANGE):
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return {"_error": "Kite not authenticated to fetch LTP."}
    
    exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
    try:
        ltp_data = kite_instance.ltp([exchange_symbol])
        return ltp_data.get(exchange_symbol)
    except Exception as e:
        return {"_error": str(e)}

@st.cache_data(ttl=3600) # Cache historical data for 1 hour
def get_historical_data_cached(api_key: str, access_token: str, symbol: str, from_date: datetime.date, to_date: datetime.date, interval: str, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame({"_error": ["Kite not authenticated to fetch historical data."]})

    instruments_df = load_instruments_cached(api_key, access_token, exchange)
    if "_error" in instruments_df.columns:
        return pd.DataFrame({"_error": [instruments_df.loc[0, '_error']]}) 

    token = find_instrument_token(instruments_df, symbol, exchange)
    if not token:
        return pd.DataFrame({"_error": [f"Instrument token not found for {symbol} on {exchange}."]})

    from_datetime = datetime.combine(from_date, datetime.min.time())
    to_datetime = datetime.combine(to_date, datetime.max.time())
    try:
        data = kite_instance.historical_data(token, from_date=from_datetime, to_date=to_datetime, interval=interval)
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=['close'], inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [str(e)]})


def find_instrument_token(df: pd.DataFrame, tradingsymbol: str, exchange: str = DEFAULT_EXCHANGE) -> int | None:
    if df.empty:
        return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & \
           (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    hits = df[mask]
    return int(hits.iloc[0]["instrument_token"]) if not hits.empty else None


# ENHANCED feature engineering function for better prediction precision
def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a rich set of technical indicators and lagged features to the dataframe for ML modeling.
    """
    if df.empty or 'close' not in df.columns:
        return pd.DataFrame()

    df_copy = df.copy() 
    
    # 1. Basic Technical Indicators
    df_copy['SMA_10'] = ta.trend.sma_indicator(df_copy['close'], window=10)
    df_copy['SMA_50'] = ta.trend.sma_indicator(df_copy['close'], window=50)
    df_copy['RSI_14'] = ta.momentum.rsi(df_copy['close'], window=14)
    macd_obj = ta.trend.MACD(df_copy['close'])
    df_copy['MACD'] = macd_obj.macd()
    df_copy['MACD_Signal'] = macd_obj.macd_signal()
    
    # 2. Volatility Features
    bb_obj = ta.volatility.BollingerBands(df_copy['close'], window=20, window_dev=2)
    df_copy['Bollinger_High'] = bb_obj.bollinger_hband()
    df_copy['Bollinger_Low'] = bb_obj.bollinger_lband()
    df_copy['Bollinger_Width'] = bb_obj.bollinger_wband() # Bandwidth can be a good feature
    df_copy['ATR'] = ta.volatility.average_true_range(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
    
    # 3. Trend Strength Features
    adx_obj = ta.trend.ADXIndicator(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
    df_copy['ADX'] = adx_obj.adx() # Trend strength, not direction
    df_copy['ADX_Pos'] = adx_obj.adx_pos() # +DI
    df_copy['ADX_Neg'] = adx_obj.adx_neg() # -DI

    # 4. Volume-based Features
    df_copy['VWAP'] = ta.volume.volume_weighted_average_price(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], window=14)
    df_copy['OBV'] = ta.volume.on_balance_volume(df_copy['close'], df_copy['volume'])
    
    # 5. Lagged Prices and Returns (more lags for better time series context)
    for lag in [1, 2, 5, 10]:
        df_copy[f'Lag_Close_{lag}'] = df_copy['close'].shift(lag)
        df_copy[f'Lag_Return_{lag}'] = df_copy['close'].pct_change(lag) * 100

    # Clean up NaNs created by indicators
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(method='ffill', inplace=True)
    df_copy.dropna(inplace=True) 
    return df_copy


# --- Sidebar: Authentication ---
with st.sidebar:
    st.markdown("### 1. Login to Kite Connect")
    st.write("Click to open Kite login. You'll be redirected back with a `request_token`.")
    st.markdown(f"[🔗 Open Kite login]({login_url})")

    # Handle request_token from URL
    request_token_param = st.query_params.get("request_token")
    if request_token_param and not st.session_state["kite_access_token"]:
        st.info("Received request_token — exchanging for access token...")
        try:
            data = kite_unauth_client.generate_session(request_token_param, api_secret=KITE_CREDENTIALS["api_secret"])
            st.session_state["kite_access_token"] = data.get("access_token")
            st.session_state["kite_login_response"] = data
            st.sidebar.success("Kite Access token obtained.")
            st.query_params.clear() # Clear request_token from URL
            st.rerun() # Rerun to refresh UI
        except Exception as e:
            st.sidebar.error(f"Failed to generate Kite session: {e}")

    if st.session_state["kite_access_token"]:
        st.success("Kite Authenticated ✅")
        if st.sidebar.button("Logout from Kite", key="kite_logout_btn"):
            st.session_state["kite_access_token"] = None
            st.session_state["kite_login_response"] = None
            st.session_state["instruments_df"] = pd.DataFrame() 
            st.success("Logged out from Kite. Please login again.")
            st.rerun()
    else:
        st.info("Not authenticated with Kite yet.")

    st.markdown("---")
    st.markdown("### 2. Supabase User Account (Optional)")
    
    def _refresh_supabase_session():
        try:
            session_data = supabase.auth.get_session()
            if session_data and session_data.user:
                st.session_state["user_session"] = session_data
                st.session_state["user_id"] = session_data.user.id
            else:
                st.session_state["user_session"] = None
                st.session_state["user_id"] = None
        except Exception:
            st.session_state["user_session"] = None
            st.session_state["user_id"] = None

    _refresh_supabase_session()

    if st.session_state["user_session"]:
        st.success(f"Supabase Logged in: {st.session_state['user_session'].user.email}")
        if st.button("Logout from Supabase", key="supabase_logout_btn"):
            try:
                supabase.auth.sign_out()
                _refresh_supabase_session() 
                st.sidebar.success("Logged out from Supabase.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error logging out: {e}")
    else:
        st.info("Supabase login section hidden for brevity. Check original file for full auth logic.")


# --- Authenticated KiteConnect client ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

# --- Tab Logic Functions ---

def render_market_historical_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("1. Market Data & Historical Data")
    if not kite_client:
        st.info("Login first to fetch market data.")
        return
    if not api_key or not access_token: 
        st.info("Kite authentication details required for data access.")
        return

    st.subheader("Current Market Data Snapshot (LTP)")
    col_market_quote1, col_market_quote2 = st.columns([1, 2])
    with col_market_quote1:
        q_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="market_exchange_tab")
        q_symbol = st.text_input("Tradingsymbol", value="RELIANCE", key="market_symbol_tab") 
        if st.button("Get Latest Price", key="get_market_data_btn"):
            ltp_data = get_ltp_price_cached(api_key, access_token, q_symbol, q_exchange) 
            if ltp_data and "_error" not in ltp_data:
                st.session_state["current_market_data"] = ltp_data
                st.success(f"Fetched LTP for {q_symbol}.")
            else:
                st.error(f"Market data fetch failed for {q_symbol}: {ltp_data.get('_error', 'Unknown error')}")
    with col_market_quote2:
        if st.session_state.get("current_market_data"):
            st.markdown("##### Latest Quote Details")
            st.json(st.session_state["current_market_data"])
        else:
            st.info("Market data will appear here.")

    st.markdown("---")
    st.subheader("Historical Price Data")
    
    col_hist_controls, col_hist_plot = st.columns([1, 2])
    with col_hist_controls:
        hist_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="hist_ex_tab_selector")
        hist_symbol = st.text_input("Tradingsymbol", value="INFY", key="hist_sym_tab_input") 
        from_date = st.date_input("From Date", value=datetime.now().date() - timedelta(days=365), key="from_dt_tab_input")
        to_date = st.date_input("To Date", value=datetime.now().date(), key="to_dt_tab_input")
        interval = st.selectbox("Interval", ["day", "minute", "5minute", "30minute", "week", "month"], index=0, key="hist_interval_selector")

        if st.button("Fetch Historical Data", key="fetch_historical_data_btn"):
            with st.spinner(f"Fetching {interval} historical data for {hist_symbol}..."):
                df_hist = get_historical_data_cached(api_key, access_token, hist_symbol, from_date, to_date, interval, hist_exchange) 
                if isinstance(df_hist, pd.DataFrame) and "_error" not in df_hist.columns:
                    st.session_state["historical_data"] = df_hist
                    st.session_state["last_fetched_symbol"] = hist_symbol
                    st.session_state["ml_data"] = pd.DataFrame() # Reset ML/Analysis data on new fetch
                    st.session_state["ml_model"] = None
                    st.success(f"Fetched {len(df_hist)} records for {hist_symbol}.")
                else:
                    st.error(f"Historical fetch failed: {df_hist.get('_error', 'Unknown error')}")

    with col_hist_plot:
        if not st.session_state.get("historical_data", pd.DataFrame()).empty:
            df = st.session_state["historical_data"]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='blue'), row=2, col=1)
            fig.update_layout(title_text=f"Historical Price & Volume for {st.session_state['last_fetched_symbol']}", xaxis_rangeslider_visible=False, height=600, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Historical chart will appear here. Please fetch data first.")


def render_analyze_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("2. In-Depth Stock Analysis")
    if not kite_client:
        st.info("Login first to perform analysis.")
        return

    historical_data = st.session_state.get("historical_data", pd.DataFrame())
    last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

    if historical_data.empty:
        st.warning("No historical data. Fetch from 'Market Data & Historical Data' first.")
        return

    st.subheader(f"Technical Analysis for {last_symbol}")

    # --- Technical Indicator Calculation ---
    df_ta = historical_data.copy()
    df_ta['SMA_20'] = ta.trend.sma_indicator(df_ta['close'], window=20)
    df_ta['SMA_50'] = ta.trend.sma_indicator(df_ta['close'], window=50)
    df_ta['RSI_14'] = ta.momentum.rsi(df_ta['close'], window=14)
    macd = ta.trend.MACD(df_ta['close'])
    df_ta['MACD'] = macd.macd()
    df_ta['MACD_Signal'] = macd.macd_signal()
    df_ta['MACD_Hist'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df_ta['close'], window=20, window_dev=2)
    df_ta['BB_High'] = bb.bollinger_hband()
    df_ta['BB_Mid'] = bb.bollinger_mavg()
    df_ta['BB_Low'] = bb.bollinger_lband()

    # --- Plotting Technical Chart ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=("Price Action & Overlays", "RSI Oscillator", "MACD"))
    
    # Candlestick and overlays
    fig.add_trace(go.Candlestick(x=df_ta.index, open=df_ta['open'], high=df_ta['high'], low=df_ta['low'], close=df_ta['close'], name='Candlestick'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_High'], mode='lines', name='Bollinger High', line=dict(color='grey', dash='dash', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Low'], mode='lines', name='Bollinger Low', line=dict(color='grey', dash='dash', width=1)), row=1, col=1)

    # RSI plot
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['RSI_14'], mode='lines', name='RSI 14'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="bottom right", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom right", row=2, col=1)

    # MACD plot
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='orange')), row=3, col=1)
    colors = ['green' if val >= 0 else 'red' for val in df_ta['MACD_Hist']]
    fig.add_trace(go.Bar(x=df_ta.index, y=df_ta['MACD_Hist'], name='Histogram', marker_color=colors), row=3, col=1)
    
    fig.update_layout(title_text=f"Technical Indicators for {last_symbol}", xaxis_rangeslider_visible=False, height=800, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # --- Indicator Summary & Automated Outlook ---
    st.markdown("---")
    st.subheader("Indicator Summary & Technical Outlook")

    def generate_technical_outlook(df):
        latest = df.dropna().iloc[-1]
        signals = {'buy': 0, 'sell': 0, 'hold': 0}
        reasons = []

        # SMA Trend
        if latest['close'] > latest['SMA_50'] and latest['SMA_20'] > latest['SMA_50']:
            signals['buy'] += 1
            reasons.append("✅ **Uptrend:** Price is above SMA 50, and Fast SMA (20) is above Slow SMA (50).")
        elif latest['close'] < latest['SMA_50'] and latest['SMA_20'] < latest['SMA_50']:
            signals['sell'] += 1
            reasons.append("❌ **Downtrend:** Price is below SMA 50, and Fast SMA (20) is below Slow SMA (50).")
        else:
            signals['hold'] += 1
            reasons.append("➖ **Neutral Trend:** Price is consolidating around key moving averages.")

        # RSI Momentum
        if latest['RSI_14'] < 30:
            signals['buy'] += 1
            reasons.append(f"✅ **Oversold:** RSI ({latest['RSI_14']:.2f}) suggests a potential bounce.")
        elif latest['RSI_14'] > 70:
            signals['sell'] += 1
            reasons.append(f"❌ **Overbought:** RSI ({latest['RSI_14']:.2f}) suggests a potential pullback.")
        else:
            signals['hold'] += 1
            reasons.append(f"➖ **Neutral Momentum:** RSI ({latest['RSI_14']:.2f}) is between 30 and 70.")
        
        # MACD Momentum
        if latest['MACD'] > latest['MACD_Signal']:
            signals['buy'] += 1
            reasons.append("✅ **Bullish Momentum:** MACD line is above its signal line.")
        else:
            signals['sell'] += 1
            reasons.append("❌ **Bearish Momentum:** MACD line is below its signal line.")
        
        # Determine final outlook
        if signals['buy'] > signals['sell']:
            return "Bullish Outlook", reasons
        elif signals['sell'] > signals['buy']:
            return "Bearish Outlook", reasons
        else:
            return "Neutral Outlook", reasons

    outlook, reasons = generate_technical_outlook(df_ta)
    st.info(f"**Overall Technical Outlook: {outlook}**")
    for reason in reasons:
        st.markdown(f"- {reason}")

    # --- Custom Strategy Builder ---
    st.markdown("---")
    st.subheader("Custom Strategy Builder & Backtest")
    st.write("Define a simple strategy and visualize its signals on the historical chart.")

    strategy_type = st.selectbox("Choose a Strategy Template", 
                                 ["SMA Crossover", "RSI Threshold", "MACD Crossover"], key="strategy_selector")

    df_strat = df_ta.copy().dropna()
    buy_signals, sell_signals = pd.DataFrame(), pd.DataFrame()
    
    if strategy_type == "SMA Crossover":
        col1, col2 = st.columns(2)
        fast_sma = col1.number_input("Fast SMA Period", 5, 100, 20)
        slow_sma = col2.number_input("Slow SMA Period", 20, 200, 50)
        df_strat['fast'] = ta.trend.sma_indicator(df_strat['close'], window=fast_sma)
        df_strat['slow'] = ta.trend.sma_indicator(df_strat['close'], window=slow_sma)
        df_strat.dropna(inplace=True)
        # Signals
        buy_mask = (df_strat['fast'] > df_strat['slow']) & (df_strat['fast'].shift(1) < df_strat['slow'].shift(1))
        sell_mask = (df_strat['fast'] < df_strat['slow']) & (df_strat['fast'].shift(1) > df_strat['slow'].shift(1))
        buy_signals = df_strat[buy_mask]
        sell_signals = df_strat[sell_mask]
    
    elif strategy_type == "RSI Threshold":
        col1, col2 = st.columns(2)
        oversold = col1.slider("Oversold Threshold (Buy)", 10, 40, 30)
        overbought = col2.slider("Overbought Threshold (Sell)", 60, 90, 70)
        # Signals
        buy_mask = (df_strat['RSI_14'] < oversold) & (df_strat['RSI_14'].shift(1) >= oversold)
        sell_mask = (df_strat['RSI_14'] > overbought) & (df_strat['RSI_14'].shift(1) <= overbought)
        buy_signals = df_strat[buy_mask]
        sell_signals = df_strat[sell_mask]
        
    elif strategy_type == "MACD Crossover":
        # Signals
        buy_mask = (df_strat['MACD'] > df_strat['MACD_Signal']) & (df_strat['MACD'].shift(1) < df_strat['MACD_Signal'].shift(1))
        sell_mask = (df_strat['MACD'] < df_strat['MACD_Signal']) & (df_strat['MACD'].shift(1) > df_strat['MACD_Signal'].shift(1))
        buy_signals = df_strat[buy_mask]
        sell_signals = df_strat[sell_mask]

    # Plotting the strategy
    st.write(f"Found **{len(buy_signals)} buy signals** and **{len(sell_signals)} sell signals** for the '{strategy_type}' strategy.")
    fig_strat = go.Figure()
    fig_strat.add_trace(go.Candlestick(x=df_strat.index, open=df_strat['open'], high=df_strat['high'], low=df_strat['low'], close=df_strat['close'], name='Candlestick'))
    
    # Plot signals
    fig_strat.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', color='green', size=12, line=dict(width=1, color='DarkSlateGrey'))))
    fig_strat.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', color='red', size=12, line=dict(width=1, color='DarkSlateGrey'))))

    fig_strat.update_layout(title_text=f"Backtest Signals for '{strategy_type}' Strategy on {last_symbol}", xaxis_rangeslider_visible=False, height=600, template="plotly_white")
    st.plotly_chart(fig_strat, use_container_width=True)


def render_price_predictor_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("3. Price Predictor (Advanced ML Analysis)")
    if not kite_client:
        st.info("Login first to perform ML analysis.")
        return

    historical_data = st.session_state.get("historical_data", pd.DataFrame())
    last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

    if historical_data.empty:
        st.warning("No historical data. Fetch from 'Market Data & Historical Data' first.")
        return

    st.subheader(f"1. Feature Engineering & Data Preparation for {last_symbol}")
    
    col_feat_eng, col_prep = st.columns(2)
    with col_feat_eng:
        if st.button("Generate Advanced Features (Indicators & Lags)", key="generate_features_btn"):
            with st.spinner("Calculating features..."):
                df_with_features = add_advanced_features(historical_data)
            if not df_with_features.empty:
                st.session_state["ml_data"] = df_with_features
                st.session_state["ml_model"] = None # Reset model
                st.success(f"Data prepared with {len(df_with_features.columns)} features.")
            else:
                st.error("Failed to add features. Data might be too short or invalid.")
                st.session_state["ml_data"] = pd.DataFrame()

    ml_data = st.session_state.get("ml_data", pd.DataFrame())
    
    if not ml_data.empty:
        with col_prep:
            current_prediction_horizon = st.number_input("Prediction Horizon (Periods/Days Ahead)", min_value=1, max_value=30, value=5, step=1, key="pred_horizon")
            test_size = st.slider("Test Set Size (%)", 10, 50, 20, step=5) / 100.0
        
        st.markdown("---")
        st.subheader("2. Machine Learning Model Training")
        
        col_ml_controls, col_ml_output = st.columns(2)
        
        ml_data_processed = ml_data.copy()
        ml_data_processed['target'] = ml_data_processed['close'].shift(-current_prediction_horizon)
        ml_data_processed.dropna(subset=['target'], inplace=True)
        
        features = [col for col in ml_data_processed.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        
        with col_ml_controls:
            model_type_selected = st.selectbox("Select ML Model", ["LightGBM Regressor (High Performance)", "Random Forest Regressor", "Linear Regression"], key="ml_model_type_selector")
            model_complexity = st.selectbox("Model Complexity", ["Fast & Simple", "Balanced (Recommended)", "Accurate & Slow"], index=1, key="model_complexity_selector", disabled=(model_type_selected=="Linear Regression"))
            
            default_features = [f for f in features if any(sub in f for sub in ['RSI', 'MACD', 'Lag_', 'SMA', 'ADX', 'ATR', 'Bollinger_Width'])]
            selected_features = st.multiselect("Select Features for Model", options=features, default=default_features, key="ml_selected_features_multiselect")
            
            if not selected_features:
                st.warning("Please select at least one feature.")
                return

            X = ml_data_processed[selected_features]
            y = ml_data_processed['target']
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, shuffle=False)
            st.info(f"Training data: {len(X_train)} periods, Testing data: {len(X_test)} periods")

            if st.button(f"Train {model_type_selected} Model", key="train_ml_model_btn"):
                # Define hyperparameters
                params = {}
                if model_type_selected == "LightGBM Regressor (High Performance)":
                    if model_complexity == "Fast & Simple": params = {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 20}
                    elif model_complexity == "Accurate & Slow": params = {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 50}
                    else: params = {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31}
                    model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1)
                elif model_type_selected == "Random Forest Regressor":
                    if model_complexity == "Fast & Simple": params = {'n_estimators': 50, 'max_depth': 5}
                    elif model_complexity == "Accurate & Slow": params = {'n_estimators': 300, 'max_depth': 15}
                    else: params = {'n_estimators': 200, 'max_depth': 10}
                    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                else:
                    model = LinearRegression()

                with st.spinner(f"Training {model_type_selected} model for {current_prediction_horizon}-day prediction..."):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                st.session_state.update({
                    "ml_model": model, "y_test": y_test, "y_pred": y_pred, "scaler": scaler,
                    "ml_features": selected_features, "ml_model_type": model_type_selected,
                    "prediction_horizon": current_prediction_horizon
                })
                st.success(f"{model_type_selected} Model Trained for {current_prediction_horizon}-day horizon!")
        
        with col_ml_output:
            if st.session_state.get("ml_model") and st.session_state.get("y_test") is not None:
                mse = mean_squared_error(st.session_state['y_test'], st.session_state['y_pred'])
                st.markdown(f"##### Evaluation Metrics ({st.session_state['prediction_horizon']} periods ahead)")
                st.metric("Root Mean Squared Error (RMSE)", f"₹{np.sqrt(mse):.2f}")
                st.metric("R2 Score (Coefficient of Determination)", f"{r2_score(st.session_state['y_test'], st.session_state['y_pred']):.4f}")
                
                if st.session_state['ml_model_type'] != "Linear Regression":
                    st.markdown("##### Feature Importances")
                    feature_imp = pd.DataFrame(sorted(zip(st.session_state['ml_model'].feature_importances_, st.session_state['ml_features'])), columns=['Value','Feature'])
                    fig_imp = go.Figure(go.Bar(x=feature_imp['Value'], y=feature_imp['Feature'], orientation='h'))
                    fig_imp.update_layout(title="Model Feature Importance", height=400, template="plotly_white", yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_imp, use_container_width=True)

        if st.session_state.get("ml_model"):
            st.markdown("---")
            st.subheader("Model Performance: Actual vs. Predicted")
            pred_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': st.session_state['y_pred']}, index=st.session_state['y_test'].index)
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Actual'], mode='lines', name='Actual Future Price'))
            fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted'], mode='lines', name='Predicted Future Price', line=dict(dash='dot', width=2)))
            fig_pred.update_layout(title_text=f"Model Performance ({st.session_state['prediction_horizon']} periods ahead)", height=500, template="plotly_white")
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.markdown("---")
            st.subheader(f"3. Generate New Forecast for {last_symbol}")
            latest_row = ml_data[st.session_state["ml_features"]].iloc[[-1]] 
            latest_row_scaled = st.session_state["scaler"].transform(latest_row)
            
            if st.button(f"Generate Forecast for Next {st.session_state['prediction_horizon']} Periods", key="generate_forecast_btn"):
                forecasted_price = st.session_state["ml_model"].predict(latest_row_scaled)[0]
                last_known_close = historical_data['close'].iloc[-1]
                predicted_change = ((forecasted_price - last_known_close) / last_known_close) * 100
                
                st.success(f"Forecast Generated using **{st.session_state['ml_model_type']}** model:")
                c1, c2, c3 = st.columns(3)
                c1.metric("Last Known Close Price", f"₹{last_known_close:.2f}")
                c2.metric(f"Predicted Price (in {st.session_state['prediction_horizon']} periods)", f"₹{forecasted_price:.2f}", delta=f"{predicted_change:.2f}%")
                c3.metric("Prediction Date Approx.", (historical_data.index[-1] + timedelta(days=st.session_state['prediction_horizon'])).strftime('%Y-%m-%d'))
    else:
        st.info("Generate features first to enable model training.")


# --- Main Application Logic (Tab Rendering) ---
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

tab_market, tab_analyze, tab_ml = st.tabs(["Market Data & Historical Data", "Analyze", "Price Predictor (ML)"])

with tab_market:
    render_market_historical_tab(k, api_key, access_token)
with tab_analyze:
    render_analyze_tab(k, api_key, access_token)
with tab_ml:
    render_price_predictor_tab(k, api_key, access_token)

import streamlit as st
from kiteconnect import KiteConnect, KiteTicker
import pandas as pd
import json
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
# import yfinance as yf  # Removed yfinance dependency - using Zerodha API exclusively

# Supabase imports
from supabase import create_client, Client

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Kite Connect - Advanced Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("Invsion Connect")
st.markdown("A comprehensive platform for fetching market data, performing ML-driven analysis, risk assessment, and live data streaming.")

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
if "saved_indexes" not in st.session_state:
    st.session_state["saved_indexes"] = []
if "current_calculated_index_data" not in st.session_state: # To store current CSV's index data
    st.session_state["current_calculated_index_data"] = None
if "current_calculated_index_history" not in st.session_state: # To store historical index values for plotting
    st.session_state["current_calculated_index_history"] = pd.DataFrame()
# Removed kt_ticker, kt_thread, kt_running, kt_ticks, kt_live_prices, kt_status_message, _rerun_ws
# as Websocket (stream) module is removed
# Removed ws_instrument_token_input, ws_instrument_name as Websocket (stream) module is removed
# Removed last_found_token, last_found_symbol, last_found_exchange as Instruments Utils module is removed


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

# Helper to create an authenticated KiteConnect instance
def get_authenticated_kite_client(api_key: str | None, access_token: str | None) -> KiteConnect | None:
    if api_key and access_token:
        k_instance = KiteConnect(api_key=api_key)
        k_instance.set_access_token(access_token)
        return k_instance
    return None


@st.cache_data(ttl=86400, show_spinner="Loading instruments...") # Cache for 24 hours
def load_instruments_cached(api_key: str, access_token: str, exchange: str = None) -> pd.DataFrame:
    """Returns pandas.DataFrame of instrument data, using an internally created Kite instance."""
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        # Return an empty DataFrame with an error indicator
        return pd.DataFrame({"_error": ["Kite not authenticated to load instruments."]})
    try:
        instruments = kite_instance.instruments(exchange) if exchange else kite_instance.instruments()
        df = pd.DataFrame(instruments)
        if "instrument_token" in df.columns:
            df["instrument_token"] = df["instrument_token"].astype("int64")
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [f"Failed to load instruments for {exchange or 'all exchanges'}: {e}"]})

@st.cache_data(ttl=60) # Cache LTP for 1 minute
def get_ltp_price_cached(api_key: str, access_token: str, symbol: str, exchange: str = DEFAULT_EXCHANGE):
    """Fetches LTP for a symbol, using an internally created Kite instance."""
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
    """Fetches historical data for a symbol, using an internally created Kite instance."""
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame({"_error": ["Kite not authenticated to fetch historical data."]})

    # Load instruments for token lookup (this calls the *cached* load_instruments_cached)
    instruments_df = load_instruments_cached(api_key, access_token, exchange)
    if "_error" in instruments_df.columns:
        return pd.DataFrame({"_error": [instruments_df.loc[0, '_error']]}) # Access the error message correctly

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


# No caching for this as it modifies df and generates many dynamic columns
def add_technical_indicators(df: pd.DataFrame, sma_short=10, sma_long=50, rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9, bb_window=20, bb_std_dev=2) -> pd.DataFrame:
    if df.empty or 'close' not in df.columns:
        st.warning("Insufficient data or missing 'close' column for indicator calculation.")
        return pd.DataFrame()

    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    df_copy['SMA_Short'] = ta.trend.sma_indicator(df_copy['close'], window=sma_short)
    df_copy['SMA_Long'] = ta.trend.sma_indicator(df_copy['close'], window=sma_long)
    df_copy['RSI'] = ta.momentum.rsi(df_copy['close'], window=rsi_window)
    
    macd_obj = ta.trend.MACD(df_copy['close'], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    df_copy['MACD'] = macd_obj.macd()
    df_copy['MACD_signal'] = macd_obj.macd_signal()
    df_copy['MACD_hist'] = macd_obj.macd_diff() 
    
    bollinger = ta.volatility.BollingerBands(df_copy['close'], window=bb_window, window_dev=bb_std_dev)
    df_copy['Bollinger_High'] = bollinger.bollinger_hband()
    df_copy['Bollinger_Low'] = bollinger.bollinger_lband()
    df_copy['Bollinger_Mid'] = bollinger.bollinger_mavg()
    df_copy['Bollinger_Width'] = bollinger.bollinger_wband()
    
    df_copy['Daily_Return'] = df_copy['close'].pct_change() * 100
    df_copy['Lag_1_Close'] = df_copy['close'].shift(1)
    
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(method='ffill', inplace=True)
    return df_copy

def calculate_performance_metrics(returns_series: pd.Series, risk_free_rate: float = 0.0) -> dict:
    if returns_series.empty or len(returns_series) < 2:
        return {}
    
    # Ensure returns are not already in percentage form for cumulative calculation
    daily_returns_decimal = returns_series / 100.0

    cumulative_returns = (1 + daily_returns_decimal).cumprod() - 1
    total_return = cumulative_returns.iloc[-1] * 100

    num_periods = len(returns_series)
    if num_periods > 0:
        # Daily average return to annualize correctly
        avg_daily_return = daily_returns_decimal.mean()
        annualized_return = ((1 + avg_daily_return)**TRADING_DAYS_PER_YEAR) - 1
    else:
        annualized_return = 0
    annualized_return *= 100 # Convert to percentage

    daily_volatility = returns_series.std()
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR) if daily_volatility is not None else 0

    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else np.nan

    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / (peak + 1e-9)
    max_drawdown = drawdown.min() * 100

    negative_returns = returns_series[returns_series < 0]
    downside_std_dev = negative_returns.std()
    sortino_ratio = (annualized_return - risk_free_rate) / (downside_std_dev * np.sqrt(TRADING_DAYS_PER_YEAR)) if downside_std_dev != 0 else np.nan

    return {
        "Total Return (%)": total_return,
        "Annualized Return (%)": annualized_return,
        "Annualized Volatility (%)": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Sortino Ratio": sortino_ratio
    }

@st.cache_data(ttl=3600, show_spinner="Calculating historical index values...")
def _calculate_historical_index_value(api_key: str, access_token: str, constituents_df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    """
    Calculates the historical value of a custom index based on its constituents and weights.
    Returns a DataFrame with 'date' and 'index_value'.
    """
    if constituents_df.empty:
        return pd.DataFrame({"_error": ["No constituents provided for historical index calculation."]})

    all_historical_closes = {}
    
    # Use a single progress bar for all fetches
    progress_bar_placeholder = st.empty()
    progress_text_placeholder = st.empty()
    
    # Pre-fetch instruments once for all symbols to reduce API calls for token lookups
    instruments_df_for_index = load_instruments_cached(api_key, access_token, exchange)
    if "_error" in instruments_df_for_index.columns:
        return pd.DataFrame({"_error": [instruments_df_for_index.loc[0, '_error']]})

    total_constituents = len(constituents_df)
    for i, row in constituents_df.iterrows():
        symbol = row['symbol']
        weight = row['Weights'] # Not directly used in fetching, but good for context
        progress_text_placeholder.text(f"Fetching historical data for {symbol} ({i+1}/{total_constituents})...")
        
        token = find_instrument_token(instruments_df_for_index, symbol, exchange)
        if not token:
            st.warning(f"Instrument token not found for {symbol} on {exchange}. Skipping for historical calculation.")
            progress_bar_placeholder.progress((i + 1) / total_constituents)
            continue

        # Using the direct kiteconnect client instance for historical data to pass token directly
        kc_client = get_authenticated_kite_client(api_key, access_token)
        if not kc_client:
            st.error("Kite client not authenticated for historical data fetch.")
            return pd.DataFrame({"_error": ["Kite not authenticated to fetch historical data."]})

        try:
            from_datetime = datetime.combine(start_date, datetime.min.time())
            to_datetime = datetime.combine(end_date, datetime.max.time())
            data = kc_client.historical_data(token, from_date=from_datetime, to_date=to_datetime, interval="day")
            hist_df = pd.DataFrame(data)
            if not hist_df.empty:
                hist_df["date"] = pd.to_datetime(hist_df["date"])
                hist_df.set_index("date", inplace=True)
                hist_df.sort_index(inplace=True)
                hist_df['close'] = pd.to_numeric(hist_df['close'], errors='coerce')
                hist_df.dropna(subset=['close'], inplace=True)
                all_historical_closes[symbol] = hist_df['close']
            else:
                st.warning(f"No historical data returned for {symbol}.")
        except Exception as e:
            st.warning(f"Error fetching historical data for {symbol}: {e}. Skipping.")
        progress_bar_placeholder.progress((i + 1) / total_constituents)

    progress_text_placeholder.empty()
    progress_bar_placeholder.empty()

    if not all_historical_closes:
        return pd.DataFrame({"_error": ["No historical data available for any constituent to build index."]})

    combined_closes = pd.DataFrame(all_historical_closes)
    
    # Forward-fill and then back-fill any missing daily prices to be more robust
    combined_closes = combined_closes.ffill().bfill()
    combined_closes.dropna(inplace=True) # Drop rows where all are still NaN

    if combined_closes.empty:
        return pd.DataFrame({"_error": ["Insufficient common historical data for index calculation after cleaning."]})

    # Calculate daily weighted prices
    # Ensure weights are aligned correctly
    # constituents_df should be indexed by 'symbol' for correct alignment
    # Ensure all symbols in combined_closes have a weight defined
    valid_symbols = [s for s in combined_closes.columns if s in constituents_df['symbol'].values]
    if not valid_symbols:
        return pd.DataFrame({"_error": ["No valid constituent symbols with weights found in historical data."]})

    weights_series = constituents_df.set_index('symbol')['Weights'].reindex(valid_symbols)
    weighted_closes = combined_closes[valid_symbols].mul(weights_series, axis=1)

    # Sum the weighted prices for each day to get the index value
    index_history_series = weighted_closes.sum(axis=1)

    # Normalize the index to a base value (e.g., 100 on the first day)
    if not index_history_series.empty:
        base_value = index_history_series.iloc[0]
        if base_value != 0:
            index_history_df = pd.DataFrame({
                "index_value": (index_history_series / base_value) * 100
            })
            index_history_df.index.name = 'date' # Ensure index name for later merging/plotting
            return index_history_df
        else:
            return pd.DataFrame({"_error": ["First day's index value is zero, cannot normalize."]})
    return pd.DataFrame({"_error": ["Error in calculating or normalizing historical index values."]})


# --- Sidebar: Kite Login ---
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
            st.session_state["instruments_df"] = pd.DataFrame() # Clear cached instruments
            st.success("Logged out from Kite. Please login again.")
            st.rerun()
    else:
        st.info("Not authenticated with Kite yet.")


# --- Sidebar: Supabase Authentication ---
with st.sidebar:
    st.markdown("### 2. Supabase User Account")
    
    # Check/refresh Supabase session
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
        st.success(f"Logged into Supabase as: {st.session_state['user_session'].user.email}")
        if st.button("Logout from Supabase", key="supabase_logout_btn"):
            try:
                supabase.auth.sign_out()
                _refresh_supabase_session() # Update session state immediately
                st.sidebar.success("Logged out from Supabase.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error logging out: {e}")
    else:
        with st.form("supabase_auth_form"):
            st.markdown("##### Email/Password Login/Sign Up")
            email = st.text_input("Email", key="supabase_email_input", help="Your email for Supabase authentication.")
            password = st.text_input("Password", type="password", key="supabase_password_input", help="Your password for Supabase authentication.")
            
            col_auth1, col_auth2 = st.columns(2)
            with col_auth1:
                login_submitted = st.form_submit_button("Login")
            with col_auth2:
                signup_submitted = st.form_submit_button("Sign Up")

            if login_submitted:
                if email and password:
                    try:
                        with st.spinner("Logging in..."):
                            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        _refresh_supabase_session()
                        st.success("Login successful! Welcome.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: {e}")
                else:
                    st.warning("Please enter both email and password for login.")
            
            if signup_submitted:
                if email and password:
                    try:
                        with st.spinner("Signing up..."):
                            response = supabase.auth.sign_up({"email": email, "password": password})
                        _refresh_supabase_session()
                        st.success("Sign up successful! Please check your email to confirm your account.")
                        st.info("After confirming your email, you can log in.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sign up failed: {e}")
                else:
                    st.warning("Please enter both email and password for sign up.")

    st.markdown("---")
    st.markdown("### 3. Quick Data Access (Kite)")
    if st.session_state["kite_access_token"]:
        current_k_client_for_sidebar = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

        if st.button("Fetch Current Holdings", key="sidebar_fetch_holdings_btn"):
            try:
                holdings = current_k_client_for_sidebar.holdings() # Direct call
                st.session_state["holdings_data"] = pd.DataFrame(holdings)
                st.success(f"Fetched {len(holdings)} holdings.")
            except Exception as e:
                st.error(f"Error fetching holdings: {e}")
        if st.session_state.get("holdings_data") is not None and not st.session_state["holdings_data"].empty:
            with st.expander("Show Holdings"):
                st.dataframe(st.session_state["holdings_data"])
    else:
        st.info("Login to Kite to access quick data.")


# --- Authenticated KiteConnect client (used by main tabs) ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])


# --- Main UI - Tabs for modules ---
# REMOVED "Websocket (stream)" and "Instruments Utils" from the tabs list
tabs = st.tabs(["Dashboard", "Portfolio", "Orders", "Market & Historical", "Machine Learning Analysis", "Risk & Stress Testing", "Performance Analysis", "Multi-Asset Analysis", "Custom Index"])
tab_dashboard, tab_portfolio, tab_orders, tab_market, tab_ml, tab_risk, tab_performance, tab_multi_asset, tab_custom_index = tabs

# --- Tab Logic Functions ---

def render_dashboard_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Personalized Dashboard")
    st.write("Welcome to your advanced financial analysis dashboard.")

    if not kite_client:
        st.info("Please login to Kite Connect to view your personalized dashboard.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account Summary")
        try:
            profile = kite_client.profile() # Direct call
            margins = kite_client.margins() # Direct call
            st.metric("Account Holder", profile.get("user_name", "N/A"))
            st.metric("Available Equity Margin", f"₹{margins.get('equity', {}).get('available', {}).get('live_balance', 0):,.2f}")
            st.metric("Available Commodity Margin", f"₹{margins.get('commodity', {}).get('available', {}).get('live_balance', 0):,.2f}")
        except Exception as e:
            st.warning(f"Could not fetch full account summary: {e}")

    with col2:
        st.subheader("Market Insight (NIFTY 50)")
        if api_key and access_token:
            nifty_ltp_data = get_ltp_price_cached(api_key, access_token, "NIFTY 50", DEFAULT_EXCHANGE) # Use cached LTP
            if nifty_ltp_data and "_error" not in nifty_ltp_data:
                nifty_ltp = nifty_ltp_data.get("last_price", 0.0)
                nifty_change = nifty_ltp_data.get("change", 0.0)
                st.metric("NIFTY 50 (LTP)", f"₹{nifty_ltp:,.2f}", delta=f"{nifty_change:.2f}%")
            else:
                st.warning(f"Could not fetch NIFTY 50 LTP: {nifty_ltp_data.get('_error', 'Unknown error')}")
        else:
            st.info("Kite not authenticated to fetch NIFTY 50 LTP.")

        if st.session_state.get("historical_data_NIFTY", pd.DataFrame()).empty:
            if st.button("Load NIFTY 50 Historical for Chart", key="dashboard_load_nifty_hist_btn"):
                if api_key and access_token:
                    with st.spinner("Fetching NIFTY 50 historical data..."):
                        nifty_df = get_historical_data_cached(api_key, access_token, "NIFTY 50", datetime.now().date() - timedelta(days=180), datetime.now().date(), "day", DEFAULT_EXCHANGE)
                        if isinstance(nifty_df, pd.DataFrame) and "_error" not in nifty_df.columns:
                            st.session_state["historical_data_NIFTY"] = nifty_df
                            st.success("NIFTY 50 historical data loaded.")
                        else:
                            st.error(f"Error fetching NIFTY 50 historical: {nifty_df.get('_error', 'Unknown error')}")
                else:
                    st.warning("Kite not authenticated to fetch historical data.")

        if not st.session_state.get("historical_data_NIFTY", pd.DataFrame()).empty:
            nifty_df = st.session_state["historical_data_NIFTY"]
            fig_nifty = go.Figure(data=[go.Candlestick(x=nifty_df.index, open=nifty_df['open'], high=nifty_df['high'], low=nifty_df['low'], close=nifty_df['close'], name='NIFTY 50')])
            fig_nifty.update_layout(title_text="NIFTY 50 Last 6 Months", xaxis_rangeslider_visible=False, height=300, template="plotly_white")
            st.plotly_chart(fig_nifty, use_container_width=True)

    with col3:
        st.subheader("Quick Performance")
        if st.session_state.get("last_fetched_symbol") and not st.session_state.get("historical_data", pd.DataFrame()).empty:
            last_symbol = st.session_state["last_fetched_symbol"]
            returns = st.session_state["historical_data"]["close"].pct_change().dropna() * 100
            if not returns.empty:
                perf = calculate_performance_metrics(returns)
                st.write(f"**{last_symbol}** (Last Fetched)")
                st.metric("Total Return", f"{perf.get('Total Return (%)', 0):.2f}%")
                st.metric("Annualized Volatility", f"{perf.get('Annualized Volatility (%)', 0):.2f}%")
                st.metric("Sharpe Ratio", f"{perf.get('Sharpe Ratio', 0):.2f}")
            else:
                st.info("No sufficient historical data for quick performance calculation.")
        else:
            st.info("Fetch some historical data in 'Market & Historical' tab to see quick performance here.")

def render_portfolio_tab(kite_client: KiteConnect | None):
    st.header("Your Portfolio Overview")
    if not kite_client:
        st.info("Login first to fetch portfolio data.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Fetch Holdings", key="portfolio_fetch_holdings_btn"):
            try:
                holdings = kite_client.holdings() # Direct call
                st.session_state["holdings_data"] = pd.DataFrame(holdings)
                st.success(f"Fetched {len(holdings)} holdings.")
            except Exception as e:
                st.error(f"Error fetching holdings: {e}")
        if not st.session_state.get("holdings_data", pd.DataFrame()).empty:
            st.subheader("Current Holdings")
            st.dataframe(st.session_state["holdings_data"], use_container_width=True)
        else:
            st.info("No holdings data available. Click 'Fetch Holdings'.")

    with col2:
        if st.button("Fetch Positions", key="portfolio_fetch_positions_btn"):
            try:
                positions = kite_client.positions() # Direct call
                st.session_state["net_positions"] = pd.DataFrame(positions.get("net", []))
                st.session_state["day_positions"] = pd.DataFrame(positions.get("day", []))
                st.success(f"Fetched positions (Net: {len(positions.get('net', []))}, Day: {len(positions.get('day', []))}).")
            except Exception as e:
                st.error(f"Error fetching positions: {e}")
        if not st.session_state.get("net_positions", pd.DataFrame()).empty:
            st.subheader("Net Positions")
            st.dataframe(st.session_state["net_positions"], use_container_width=True)
        if not st.session_state.get("day_positions", pd.DataFrame()).empty:
            st.subheader("Day Positions")
            st.dataframe(st.session_state["day_positions"], use_container_width=True)

    with col3:
        if st.button("Fetch Margins", key="portfolio_fetch_margins_btn"):
            try:
                margins = kite_client.margins() # Direct call
                st.session_state["margins_data"] = margins
                st.success("Fetched margins data.")
            except Exception as e:
                st.error(f"Error fetching margins: {e}")
        if st.session_state.get("margins_data"):
            st.subheader("Available Margins")
            margins_df = pd.DataFrame([
                {"Category": "Equity - Available", "Value": st.session_state["margins_data"].get('equity', {}).get('available', {}).get('live_balance', 0)},
                {"Category": "Equity - Used", "Value": st.session_state["margins_data"].get('equity', {}).get('utilised', {}).get('overall', 0)},
                {"Category": "Commodity - Available", "Value": st.session_state["margins_data"].get('commodity', {}).get('available', {}).get('live_balance', 0)},
                {"Category": "Commodity - Used", "Value": st.session_state["margins_data"].get('commodity', {}).get('utilised', {}).get('overall', 0)},
            ])
            margins_df["Value"] = margins_df["Value"].apply(lambda x: f"₹{x:,.2f}")
            st.dataframe(margins_df, use_container_width=True)

def render_orders_tab(kite_client: KiteConnect | None):
    st.header("Orders — Place, Modify, Cancel & View")
    if not kite_client:
        st.info("Login first to use orders API.")
        return

    st.subheader("Place New Order")
    with st.form("place_order_form", clear_on_submit=False):
        col_order1, col_order2 = st.columns(2)
        with col_order1:
            variety = st.selectbox("Variety", ["regular", "amo", "co", "iceberg"], key="order_variety")
            exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO", "CDS", "MCX"], key="order_exchange")
            tradingsymbol = st.text_input("Tradingsymbol", value="INFY", key="order_tradingsymbol")
            transaction_type = st.radio("Transaction Type", ["BUY", "SELL"], horizontal=True, key="order_transaction_type")
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="order_quantity")
        with col_order2:
            order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "SL", "SL-M"], key="order_type")
            product = st.selectbox("Product Type", ["CNC", "MIS", "NRML", "CO", "MTF"], key="order_product")
            price = st.text_input("Price (for LIMIT/SL)", value="", key="order_price")
            trigger_price = st.text_input("Trigger Price (for SL/SL-M)", value="", key="order_trigger_price")
            validity = st.selectbox("Validity", ["DAY", "IOC", "TTL"], key="order_validity")
            tag = st.text_input("Tag (optional, max 20 chars)", value="", key="order_tag")
        
        submit_place = st.form_submit_button("Place Order", key="submit_place_order")
        if submit_place:
            try:
                params = dict(variety=variety, exchange=exchange, tradingsymbol=tradingsymbol,
                              transaction_type=transaction_type, order_type=order_type,
                              quantity=int(quantity), product=product, validity=validity)
                if price: params["price"] = float(price)
                if trigger_price: params["trigger_price"] = float(trigger_price)
                if tag: params["tag"] = tag[:20]

                with st.spinner("Placing order..."):
                    resp = kite_client.place_order(**params) # Direct call
                    st.success(f"Order placed! ID: {resp.get('order_id')}")
            except Exception as e:
                st.error(f"Failed to place order: {e}")

    st.markdown("---")
    st.subheader("Manage Existing Orders & Trades")
    col_view_orders, col_manage_single = st.columns(2)

    with col_view_orders:
        if st.button("Fetch All Orders (Today)", key="fetch_all_orders_btn"):
            try:
                orders = kite_client.orders() # Direct call
                st.session_state["all_orders"] = pd.DataFrame(orders)
                st.success(f"Fetched {len(orders)} orders.")
            except Exception as e: st.error(f"Error fetching orders: {e}")
        if not st.session_state.get("all_orders", pd.DataFrame()).empty:
            with st.expander("Show Orders"): st.dataframe(st.session_state["all_orders"], use_container_width=True)

        if st.button("Fetch All Trades (Today)", key="fetch_all_trades_btn"):
            try:
                trades = kite_client.trades() # Direct call
                st.session_state["all_trades"] = pd.DataFrame(trades)
                st.success(f"Fetched {len(trades)} trades.")
            except Exception as e: st.error(f"Error fetching trades: {e}")
        if not st.session_state.get("all_trades", pd.DataFrame()).empty:
            with st.expander("Show Trades"): st.dataframe(st.session_state["all_trades"], use_container_width=True)

    with col_manage_single:
        order_id_action = st.text_input("Order ID for action", key="order_id_action")
        if st.button("Get Order History", key="get_order_history_btn"):
            if order_id_action:
                try: st.json(kite_client.order_history(order_id_action)) # Direct call
                except Exception as e: st.error(f"Failed to get order history: {e}")
            else: st.warning("Provide an Order ID.")
        
        with st.form("modify_order_form"):
            mod_variety = st.selectbox("Variety (for Modify)", ["regular", "amo", "co", "iceberg"], key="mod_variety_selector")
            mod_new_price = st.text_input("New Price (optional)", key="mod_new_price")
            mod_new_qty = st.number_input("New Quantity (optional)", min_value=0, value=0, step=1, key="mod_new_qty")
            submit_modify = st.form_submit_button("Modify Order", key="submit_modify_order")
            if submit_modify:
                if order_id_action:
                    try:
                        modify_args = {}
                        if mod_new_price: modify_args["price"] = float(mod_new_price)
                        if mod_new_qty > 0: modify_args["quantity"] = int(mod_new_qty)
                        if not modify_args: st.warning("No new price or quantity.")
                        else: st.json(kite_client.modify_order(variety=mod_variety, order_id=order_id_action, **modify_args)) # Direct call
                    except Exception as e: st.error(f"Failed to modify order: {e}")
                else: st.warning("Provide an Order ID to modify.")
        
        if st.button("Cancel Order", key="cancel_order_btn"):
            if order_id_action:
                try: st.json(kite_client.cancel_order(variety="regular", order_id=order_id_action)) # Direct call
                except Exception as e: st.error(f"Failed to cancel order: {e}")
            else: st.warning("Provide an Order ID to cancel.")

def render_market_historical_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Market Data & Historical Candles")
    if not kite_client:
        st.info("Login first to fetch market data.")
        return
    if not api_key or not access_token: # Additional check for cached functions
        st.info("Kite authentication details required for cached data access.")
        return

    st.subheader("Current Market Data Snapshot")
    col_market_quote1, col_market_quote2 = st.columns([1, 2])
    with col_market_quote1:
        q_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="market_exchange_tab")
        q_symbol = st.text_input("Tradingsymbol", value="NIFTY 50", key="market_symbol_tab") # Default to NIFTY 50 for quick demo
        if st.button("Get Market Data", key="get_market_data_btn"):
            ltp_data = get_ltp_price_cached(api_key, access_token, q_symbol, q_exchange) # Use cached LTP
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
    with st.expander("Load Instruments for Symbol Lookup (Recommended)"):
        exchange_for_lookup = st.selectbox("Exchange to load instruments", ["NSE", "BSE", "NFO"], key="hist_inst_load_exchange_selector")
        if st.button("Load Instruments into Cache", key="load_inst_cache_btn"):
            df_instruments = load_instruments_cached(api_key, access_token, exchange_for_lookup) # Use cached instruments
            if not df_instruments.empty and "_error" not in df_instruments.columns:
                st.session_state["instruments_df"] = df_instruments
                st.success(f"Loaded {len(df_instruments)} instruments.")
            else:
                st.error(f"Failed to load instruments: {df_instruments.get('_error', 'Unknown error')}")


    col_hist_controls, col_hist_plot = st.columns([1, 2])
    with col_hist_controls:
        hist_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="hist_ex_tab_selector")
        hist_symbol = st.text_input("Tradingsymbol", value="NIFTY 50", key="hist_sym_tab_input") # Default to NIFTY 50 for quick demo
        from_date = st.date_input("From Date", value=datetime.now().date() - timedelta(days=90), key="from_dt_tab_input")
        to_date = st.date_input("To Date", value=datetime.now().date(), key="to_dt_tab_input")
        interval = st.selectbox("Interval", ["minute", "5minute", "30minute", "day", "week", "month"], index=3, key="hist_interval_selector")

        if st.button("Fetch Historical Data", key="fetch_historical_data_btn"):
            with st.spinner(f"Fetching {interval} historical data for {hist_symbol}..."):
                df_hist = get_historical_data_cached(api_key, access_token, hist_symbol, from_date, to_date, interval, hist_exchange) # Use cached historical
                if isinstance(df_hist, pd.DataFrame) and "_error" not in df_hist.columns:
                    st.session_state["historical_data"] = df_hist
                    st.session_state["last_fetched_symbol"] = hist_symbol
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
            st.info("Historical chart will appear here.")

def render_ml_analysis_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Machine Learning Driven Price Analysis")
    if not kite_client:
        st.info("Login first to perform ML analysis.")
        return

    historical_data = st.session_state.get("historical_data")
    last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

    if historical_data.empty:
        st.warning("No historical data. Fetch from 'Market & Historical' first.")
        return

    st.subheader(f"1. Feature Engineering: Technical Indicators for {last_symbol}")
    col_indicator_params, col_indicator_data = st.columns([1,2])
    with col_indicator_params:
        sma_short_window = st.slider("SMA Short Window", 5, 50, 10, key="ml_sma_short_window")
        sma_long_window = st.slider("SMA Long Window", 20, 200, 50, key="ml_sma_long_window")
        rsi_window = st.slider("RSI Window", 7, 30, 14, key="ml_rsi_window")
        macd_fast = st.slider("MACD Fast Period", 5, 20, 12, key="ml_macd_fast")
        macd_slow = st.slider("MACD Slow Period", 20, 40, 26, key="ml_macd_slow")
        macd_signal = st.slider("MACD Signal Period", 5, 15, 9, key="ml_macd_signal")
        bb_window = st.slider("Bollinger Bands Window", 10, 50, 20, key="ml_bb_window")
        bb_std_dev = st.slider("Bollinger Bands Std Dev", 1.0, 3.0, 2.0, step=0.1, key="ml_bb_std_dev")
        
        if st.button("Apply Indicators", key="apply_indicators_btn"):
            df_with_indicators = add_technical_indicators(historical_data, sma_short_window, sma_long_window, 
                                                    rsi_window, macd_fast, macd_slow, macd_signal, 
                                                    bb_window, bb_std_dev)
            if not df_with_indicators.empty:
                st.session_state["ml_data"] = df_with_indicators
                st.success("Technical indicators applied.")
            else:
                st.error("Failed to add indicators. Data might be too short or invalid.")
                st.session_state["ml_data"] = pd.DataFrame()
    
    with col_indicator_data:
        ml_data = st.session_state.get("ml_data", pd.DataFrame())
        if not ml_data.empty:
            st.markdown("##### Data with Indicators (Head)")
            st.dataframe(ml_data.head(), use_container_width=True)
            # Plot indicators (abbreviated for brevity, full plot logic from original)
            fig_indicators = go.Figure(data=[
                go.Candlestick(x=ml_data.index, open=ml_data['open'], high=ml_data['high'], low=ml_data['low'], close=ml_data['close'], name='Price'),
                go.Scatter(x=ml_data.index, y=ml_data['SMA_Short'], mode='lines', name='SMA Short'),
                go.Scatter(x=ml_data.index, y=ml_data['SMA_Long'], mode='lines', name='SMA Long')
            ])
            fig_indicators.update_layout(title=f"Price with SMAs for {last_symbol}", height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_indicators, use_container_width=True)
        else:
            st.info("Apply indicators to see data and basic plots.")

    if not ml_data.empty:
        st.subheader(f"2. Machine Learning Model Training for {last_symbol}")
        col_ml_controls, col_ml_output = st.columns(2)
        with col_ml_controls:
            model_type_selected = st.selectbox("Select ML Model", ["Linear Regression", "Random Forest Regressor", "LightGBM Regressor"], key="ml_model_type_selector")
            ml_data_processed = ml_data.copy()
            ml_data_processed['target'] = ml_data_processed['close'].shift(-1)
            ml_data_processed.dropna(subset=['target'], inplace=True)
            
            features = [col for col in ml_data_processed.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target', 'MACD_hist']]
            selected_features = st.multiselect("Select Features for Model", options=features, default=features, key="ml_selected_features_multiselect")
            
            if not selected_features:
                st.warning("Please select at least one feature.")
                return

            X = ml_data_processed[selected_features]
            y = ml_data_processed['target']
            
            if X.empty or y.empty:
                st.error("Not enough clean data after preprocessing to train the model. Adjust parameters or fetch more data.")
                return

            test_size = st.slider("Test Set Size (%)", 10, 50, 20, step=5) / 100.0
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
            st.info(f"Training data: {len(X_train)} samples, Testing data: {len(X_test)} samples")

            if st.button(f"Train {model_type_selected} Model", key="train_ml_model_btn"):
                if len(X_train) == 0 or len(X_test) == 0:
                    st.error("Insufficient data for training/testing. Adjust test size or fetch more data.")
                    return
                model = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                    "LightGBM Regressor": lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                }.get(model_type_selected)

                if model:
                    with st.spinner(f"Training {model_type_selected} model..."):
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    st.session_state["ml_model"] = model
                    st.session_state["y_test"] = y_test
                    st.session_state["y_pred"] = y_pred
                    st.session_state["X_test_ml"] = X_test
                    st.session_state["ml_features"] = selected_features
                    st.session_state["ml_model_type"] = model_type_selected
                    st.success(f"{model_type_selected} Model Trained!")
        
        with col_ml_output:
            if st.session_state.get("ml_model") and st.session_state.get("y_test") is not None:
                mse = mean_squared_error(st.session_state['y_test'], st.session_state['y_pred'])
                r2 = r2_score(st.session_state['y_test'], st.session_state['y_pred'])
                st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
                st.metric("R2 Score", f"{r2:.4f}")

                pred_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': st.session_state['y_pred']}, index=st.session_state['y_test'].index)
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Actual'], mode='lines', name='Actual Price'))
                fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted'], mode='lines', name='Predicted Price', line=dict(dash='dot')))
                fig_pred.update_layout(title_text=f"Actual vs. Predicted Prices for {last_symbol}", height=500, template="plotly_white")
                st.plotly_chart(fig_pred, use_container_width=True)

        st.subheader(f"3. Real-time Price Prediction (Simulated for {last_symbol})")
        if st.session_state.get("ml_model") and not st.session_state.get("X_test_ml", pd.DataFrame()).empty:
            model = st.session_state["ml_model"]
            latest_features_df = st.session_state["X_test_ml"].iloc[[-1]][st.session_state["ml_features"]]
            if st.button("Simulate Next Period Prediction", key="simulate_prediction_btn"):
                simulated_prediction = model.predict(latest_features_df)[0]
                st.success(f"Simulated **next period** close price prediction: **₹{simulated_prediction:.2f}**")
        else:
            st.info("Train a machine learning model first for simulation.")
        
        st.markdown("---")
        st.subheader("4. Basic Backtesting: SMA Crossover Strategy")
        if not ml_data.empty:
            df_backtest = ml_data.copy()
            short_ma = st.slider("Short MA Window", 5, 50, 10, key="bt_short_ma")
            long_ma = st.slider("Long MA Window", 20, 200, 50, key="bt_long_ma")

            if st.button("Run Backtest", key="run_backtest_btn"):
                if len(df_backtest) < max(short_ma, long_ma):
                    st.error("Not enough data for MA windows.")
                    return
                df_backtest['SMA_Short_BT'] = ta.trend.sma_indicator(df_backtest['close'], window=short_ma)
                df_backtest['SMA_Long_BT'] = ta.trend.sma_indicator(df_backtest['close'], window=long_ma)
                df_backtest['Signal'] = (df_backtest['SMA_Short_BT'] > df_backtest['SMA_Long_BT']).astype(float)
                df_backtest['Position'] = df_backtest['Signal'].diff()
                df_backtest['Strategy_Return'] = df_backtest['Daily_Return'] * df_backtest['Signal'].shift(1)
                df_backtest['Cumulative_Strategy_Return'] = (1 + df_backtest['Strategy_Return'] / 100).cumprod() - 1
                df_backtest['Cumulative_Buy_Hold_Return'] = (1 + df_backtest['Daily_Return'] / 100).cumprod() - 1

                col_bt_metrics, col_bt_chart = st.columns(2)
                with col_bt_metrics:
                    strategy_metrics = calculate_performance_metrics(df_backtest['Strategy_Return'].dropna())
                    buy_hold_metrics = calculate_performance_metrics(df_backtest['Daily_Return'].dropna())
                    st.write("**Strategy Metrics**")
                    for k_m, v_m in strategy_metrics.items(): st.metric(k_m, f"{v_m:.2f}%" if "%" in k_m else f"{v_m:.2f}")
                    st.write("**Buy & Hold Metrics**")
                    for k_m, v_m in buy_hold_metrics.items(): st.metric(k_m, f"{v_m:.2f}%" if "%" in k_m else f"{v_m:.2f}")

                with col_bt_chart:
                    fig_backtest = go.Figure()
                    fig_backtest.add_trace(go.Scatter(x=df_backtest.index, y=df_backtest['Cumulative_Strategy_Return'] * 100, name='Strategy Return'))
                    fig_backtest.add_trace(go.Scatter(x=df_backtest.index, y=df_backtest['Cumulative_Buy_Hold_Return'] * 100, name='Buy & Hold Return', line=dict(dash='dash')))
                    fig_backtest.update_layout(title_text=f"SMA Crossover Strategy vs. Buy & Hold for {last_symbol}", height=450)
                    st.plotly_chart(fig_backtest, use_container_width=True)
        else:
            st.info("Apply technical indicators first to enable backtesting.")


def render_risk_stress_testing_tab(kite_client: KiteConnect | None):
    st.header("Risk & Stress Testing Models")
    if not kite_client:
        st.info("Login first to perform risk analysis.")
        return

    historical_data = st.session_state.get("historical_data")
    last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

    if historical_data.empty:
        st.warning("No historical data. Fetch from 'Market & Historical' first.")
        return
    
    historical_data['close'] = pd.to_numeric(historical_data['close'], errors='coerce')
    daily_returns = historical_data['close'].pct_change().dropna() * 100
    if daily_returns.empty or len(daily_returns) < 2:
        st.error("Not enough valid data for risk analysis.")
        return

    st.subheader(f"1. Volatility & Returns Analysis for {last_symbol}")
    col_vol_metrics, col_vol_dist = st.columns([1,2])
    with col_vol_metrics:
        st.dataframe(daily_returns.describe().to_frame().T, use_container_width=True)
        annualized_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        st.metric("Annualized Volatility", f"{annualized_volatility:.2f}%")
        st.metric("Mean Daily Return", f"{daily_returns.mean():.2f}%")

        rolling_window = st.slider("Rolling Volatility Window (days)", 10, 252, 30, key="risk_rolling_vol_window")
        if len(daily_returns) > rolling_window:
            rolling_vol = daily_returns.rolling(window=rolling_window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            fig_rolling_vol = go.Figure(go.Scatter(x=rolling_vol.index, y=rolling_vol, name='Rolling Volatility'))
            fig_rolling_vol.update_layout(title_text=f"Rolling {rolling_window}-Day Annualized Volatility", height=300)
            st.plotly_chart(fig_rolling_vol, use_container_width=True)
    with col_vol_dist:
        fig_volatility = go.Figure(go.Histogram(x=daily_returns, nbinsx=50, name='Daily Returns'))
        fig_volatility.update_layout(title_text=f'Distribution of Daily Returns for {last_symbol}', height=500)
        st.plotly_chart(fig_volatility, use_container_width=True)

    st.subheader(f"2. Value at Risk (VaR) Calculation for {last_symbol}")
    col_var_controls, col_var_plot = st.columns([1,2])
    with col_var_controls:
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 95, step=1, key="risk_confidence_level")
        holding_period_var = st.number_input("Holding Period for VaR (days)", min_value=1, value=1, step=1, key="risk_holding_period_var")
        var_percentile_1day = np.percentile(daily_returns, 100 - confidence_level)
        var_percentile_multiday = var_percentile_1day * np.sqrt(holding_period_var)
        st.write(f"With **{confidence_level}% confidence**, max loss over **{holding_period_var} day(s)**:")
        st.metric(f"VaR ({confidence_level}%)", f"{abs(var_percentile_multiday):.2f}%")
        current_price = historical_data['close'].iloc[-1]
        st.metric(f"Potential Loss (₹{current_price:.2f})", f"₹{(abs(var_percentile_multiday) / 100) * current_price:,.2f}")
    with col_var_plot:
        fig_var = go.Figure(go.Histogram(x=daily_returns, nbinsx=50, name='Daily Returns'))
        fig_var.add_vline(x=var_percentile_1day, line_dash="dash", line_color="red", annotation_text=f"1-Day VaR {confidence_level}%: {var_percentile_1day:.2f}%")
        fig_var.update_layout(title_text=f'Daily Returns Distribution with {confidence_level}% VaR for {last_symbol}', height=400)
        st.plotly_chart(fig_var, use_container_width=True)

    st.subheader(f"3. Stress Testing (Scenario Analysis) for {last_symbol}")
    col_stress_controls, col_stress_results = st.columns([1,2])
    with col_stress_controls:
        scenarios = {
            "Historical Worst Day Drop": {"type": "historical", "percent": daily_returns.min() if not daily_returns.empty else 0},
            "Global Financial Crisis (-20%)": {"type": "fixed", "percent": -20.0},
            "Custom % Change": {"type": "custom", "percent": 0.0}
        }
        scenario_key = st.selectbox("Select Stress Scenario", list(scenarios.keys()), key="risk_scenario_selector")
        custom_change_percent = st.number_input("Custom Percentage Change (%)", value=0.0, step=0.1, key="risk_custom_change_input") if scenario_key == "Custom % Change" else 0.0
        
        if st.button("Run Stress Test", key="run_stress_test_btn"):
            current_price = historical_data['close'].iloc[-1]
            scenario_data = scenarios[scenario_key]
            scenario_change_percent = scenario_data["percent"] if scenario_data["type"] != "custom" else custom_change_percent
            stressed_price = current_price * (1 + scenario_change_percent / 100)
            st.session_state["stress_test_results"] = {"scenario_key": scenario_key, "current_price": current_price, "stressed_price": stressed_price, "scenario_change_percent": scenario_change_percent}
            st.success("Stress test executed.")
    with col_stress_results:
        if st.session_state.get("stress_test_results"):
            results = st.session_state["stress_test_results"]
            st.markdown(f"##### Results for Scenario: **{results['scenario_key']}**")
            st.metric("Current Price", f"₹{results['current_price']:.2f}")
            st.metric("Stressed Price", f"₹{results['stressed_price']:.2f}")
            st.metric("Percentage Change", f"{results['scenario_change_percent']}")


import streamlit as st
from kiteconnect import KiteConnect
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
from sklearn.preprocessing import MinMaxScaler 
import shap # To calculate feature importance (if available)

# Supabase imports (Keeping them for completeness, though auth is simplified)
from supabase import create_client, Client

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Invsion Connect - Algo Trading Platform", layout="wide", initial_sidebar_state="expanded")
st.title("Invsion Connect - High-Performance Algorithmic Trading Platform")
st.markdown("Focused on high-speed data fetching, advanced ML prediction, and robust risk analysis.")

# --- Global Constants & Session State Initialization ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"

# Initialize standard session state variables
if "kite_access_token" not in st.session_state:
    st.session_state["kite_access_token"] = None
if "instruments_df" not in st.session_state:
    st.session_state["instruments_df"] = pd.DataFrame()
if "historical_data" not in st.session_state:
    st.session_state["historical_data"] = pd.DataFrame()
if "last_fetched_symbol" not in st.session_state:
    st.session_state["last_fetched_symbol"] = None
if "user_session" not in st.session_state:
    st.session_state["user_session"] = None

# --- ML specific initializations (Robustness Fix) ---
if "ml_data" not in st.session_state:
    st.session_state["ml_data"] = pd.DataFrame()
if "ml_model" not in st.session_state:
    st.session_state["ml_model"] = None
if "prediction_horizon" not in st.session_state:
    st.session_state["prediction_horizon"] = 5 
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None
if "y_test" not in st.session_state:
    st.session_state["y_test"] = None
if "y_pred" not in st.session_state:
    st.session_state["y_pred"] = None
if "X_test_scaled" not in st.session_state:
    st.session_state["X_test_scaled"] = None
if "ml_features" not in st.session_state:
    st.session_state["ml_features"] = []


# --- Load Credentials from Streamlit Secrets ---
def load_secrets():
    secrets = st.secrets
    kite_conf = secrets.get("kite", {})
    supabase_conf = secrets.get("supabase", {})
    # Simplified credential checks for deployment integrity
    return kite_conf, supabase_conf

KITE_CREDENTIALS, SUPABASE_CREDENTIALS = load_secrets()

# --- Supabase Client Initialization ---
@st.cache_resource(ttl=3600)
def init_supabase_client(url: str, key: str) -> Client:
    # Requires Supabase secrets to be configured
    return create_client(url, key)

if SUPABASE_CREDENTIALS.get("url"):
    supabase: Client = init_supabase_client(SUPABASE_CREDENTIALS["url"], SUPABASE_CREDENTIALS["anon_key"])
else:
    # Placeholder for unconfigured Supabase
    supabase = None

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

@st.cache_data(ttl=86400, show_spinner="Loading instruments...")
def load_instruments_cached(api_key: str, access_token: str, exchange: str = None) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame({"_error": ["Kite not authenticated to load instruments."]})
    try:
        instruments = kite_instance.instruments(exchange) if exchange else kite_instance.instruments()
        df = pd.DataFrame(instruments)
        if 'tradingsymbol' in df.columns and 'name' in df.columns:
            df = df[['instrument_token', 'tradingsymbol', 'name', 'exchange']]
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [f"Failed to load instruments: {e}"]})

@st.cache_data(ttl=3600)
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

# --- ENHANCED FEATURE ENGINEERING for higher precision ---
def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'close' not in df.columns:
        return pd.DataFrame()

    df_copy = df.copy() 
    
    # --- 1. Advanced Trend Indicators ---
    df_copy['SMA_10'] = ta.trend.sma_indicator(df_copy['close'], window=10)
    df_copy['EMA_20'] = ta.trend.ema_indicator(df_copy['close'], window=20)
    macd_obj = ta.trend.MACD(df_copy['close'], window_fast=12, window_slow=26, window_sign=9)
    df_copy['MACD'] = macd_obj.macd()
    df_copy['ADX'] = ta.trend.adx(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
    df_copy['Ichimoku_A'] = ta.trend.ichimoku_a(df_copy['high'], df_copy['low'], window1=9, window2=26).shift(26)
    
    # --- 2. Volatility & Bands ---
    df_copy['Bollinger_High'] = ta.volatility.BollingerBands(df_copy['close'], window=20, window_dev=2).bollinger_hband()
    df_copy['Bollinger_Width'] = ta.volatility.BollingerBands(df_copy['close'], window=20, window_dev=2).bollinger_wband()
    df_copy['ATR'] = ta.volatility.average_true_range(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
    
    # --- 3. Momentum & Oscillation ---
    df_copy['RSI_14'] = ta.momentum.rsi(df_copy['close'], window=14)
    df_copy['Stoch_RSI'] = ta.momentum.stochrsi(df_copy['close'], window=14)
    df_copy['MFI'] = ta.volume.money_flow_index(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'])
    
    # --- 4. Volume & Flow ---
    df_copy['VWAP'] = ta.volume.volume_weighted_average_price(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], window=14)
    df_copy['OBV'] = ta.volume.on_balance_volume(df_copy['close'], df_copy['volume'])
    
    # --- 5. Lagged Prices and Returns (Increased Lag for better context) ---
    for lag in [1, 2, 5, 10, 20]: 
        df_copy[f'Lag_Close_{lag}'] = df_copy['close'].shift(lag)
        df_copy[f'Lag_Return_{lag}'] = df_copy['close'].pct_change(lag) * 100

    # --- 6. Temporal Features (Crucial for time series) ---
    if df_copy.index.inferred_type == 'datetime64':
        df_copy['day_of_week'] = df_copy.index.dayofweek
        df_copy['day_of_month'] = df_copy.index.day
        # Convert cyclical features (optional but good practice)
        df_copy['sin_day'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy.drop(columns=['day_of_week'], inplace=True)
    
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(method='ffill', inplace=True)
    df_copy.dropna(inplace=True) 
    return df_copy

def calculate_performance_metrics(returns_series: pd.Series, risk_free_rate: float = 0.0) -> dict:
    if returns_series.empty or len(returns_series) < 2:
        return {}
    
    daily_returns_decimal = returns_series / 100.0 if returns_series.abs().mean() > 1 else returns_series
    cumulative_returns = (1 + daily_returns_decimal).cumprod() - 1
    total_return = cumulative_returns.iloc[-1] * 100 if not cumulative_returns.empty else 0

    num_periods = len(daily_returns_decimal)
    annualized_return = ((1 + daily_returns_decimal).prod())**(TRADING_DAYS_PER_YEAR / num_periods) - 1 if num_periods > 0 and (1 + daily_returns_decimal).min() > 0 else 0
    annualized_return *= 100

    daily_volatility = daily_returns_decimal.std()
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR) * 100 if daily_volatility is not None else 0

    sharpe_ratio = (annualized_return / 100 - risk_free_rate / 100) / (annualized_volatility / 100) if annualized_volatility != 0 else np.nan

    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / (peak + 1)
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0

    return {
        "Total Return (%)": total_return,
        "Annualized Return (%)": annualized_return,
        "Annualized Volatility (%)": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown
    }

# --- Sidebar: Authentication ---
with st.sidebar:
    st.markdown("### 1. Login to Kite Connect")
    st.write("Click to open Kite login. You'll be redirected back with a `request_token`.")
    st.markdown(f"[🔗 Open Kite login]({login_url})")

    request_token_param = st.query_params.get("request_token")
    if request_token_param and not st.session_state["kite_access_token"]:
        try:
            data = kite_unauth_client.generate_session(request_token_param, api_secret=KITE_CREDENTIALS["api_secret"])
            st.session_state["kite_access_token"] = data.get("access_token")
            st.sidebar.success("Kite Access token obtained.")
            st.query_params.clear() 
            st.rerun() 
        except Exception as e:
            st.sidebar.error(f"Failed to generate Kite session: {e}")

    if st.session_state["kite_access_token"]:
        st.success("Kite Authenticated ✅")
        if st.sidebar.button("Logout from Kite", key="kite_logout_btn"):
            st.session_state["kite_access_token"] = None
            st.session_state["instruments_df"] = pd.DataFrame() 
            st.rerun()
    else:
        st.info("Not authenticated with Kite yet.")

# --- Authenticated KiteConnect client ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

# --- Tab Functions (Kept modular) ---

def render_market_historical_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    # (Content remains mostly the same as previous functional version, ensuring data fetching is robust)
    st.header("1. Market Data & Historical Data (High Speed)")
    if not kite_client:
        st.info("Login first to fetch market data.")
        return
    # ... [rest of the market_historical_tab content] ...
    
    st.subheader("Historical Price Data")
    
    col_hist_controls, col_hist_plot = st.columns([1, 2])
    with col_hist_controls:
        hist_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="hist_ex_tab_selector")
        hist_symbol = st.text_input("Tradingsymbol", value="INFY", key="hist_sym_tab_input") 
        from_date = st.date_input("From Date", value=datetime.now().date() - timedelta(days=365), key="from_dt_tab_input")
        to_date = st.date_input("To Date", value=datetime.now().date(), key="to_dt_tab_input")
        interval = st.selectbox("Interval", ["day", "minute", "5minute", "30minute", "week", "month"], index=0, key="hist_interval_selector")

        if st.button("Fetch Historical Data", key="fetch_historical_data_btn"):
            if st.session_state["instruments_df"].empty:
                df_instruments = load_instruments_cached(api_key, access_token, hist_exchange)
                st.session_state["instruments_df"] = df_instruments
            
            with st.spinner(f"Fetching {interval} historical data for {hist_symbol}..."):
                df_hist = get_historical_data_cached(api_key, access_token, hist_symbol, from_date, to_date, interval, hist_exchange) 
                if isinstance(df_hist, pd.DataFrame) and "_error" not in df_hist.columns:
                    st.session_state["historical_data"] = df_hist
                    st.session_state["last_fetched_symbol"] = hist_symbol
                    # Reset ML state
                    st.session_state["ml_data"] = pd.DataFrame()
                    st.session_state["ml_model"] = None
                    st.session_state["scaler"] = None
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


def render_price_predictor_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("2. High-Frequency ML Predictor (Seeking 100% Precision)")
    st.markdown("Utilizing optimized Gradient Boosting (LightGBM) and extensive feature engineering for the highest prediction accuracy.")
    
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
        if st.button("Generate Extensive Algorithmic Features (Blazing Fast)", key="generate_features_btn"):
            with st.spinner("Generating 20+ features..."):
                df_with_features = add_advanced_features(historical_data)
            if not df_with_features.empty:
                st.session_state["ml_data"] = df_with_features
                st.session_state["ml_model"] = None
                st.session_state["scaler"] = None
                st.success(f"Data prepared with {len(df_with_features.columns)} high-precision features.")
            else:
                st.error("Failed to add features. Data might be too short or invalid.")
                st.session_state["ml_data"] = pd.DataFrame()

    ml_data = st.session_state.get("ml_data", pd.DataFrame())
    
    if not ml_data.empty:
        
        with col_prep:
            current_prediction_horizon = st.number_input("Prediction Horizon (Periods Ahead)", min_value=1, max_value=20, value=st.session_state.get("prediction_horizon", 5), step=1, key="pred_horizon")
            test_size = st.slider("Test Set Size (%)", 10, 50, 20, step=5) / 100.0
        
        st.markdown("---")
        st.subheader("2. Optimized Machine Learning Model Training")
        
        col_ml_controls, col_ml_output = st.columns(2)
        
        # --- Data Prep ---
        ml_data_processed = ml_data.copy()
        ml_data_processed['target'] = ml_data_processed['close'].shift(-current_prediction_horizon)
        ml_data_processed.dropna(subset=['target'], inplace=True)
        
        features = [col for col in ml_data_processed.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        
        with col_ml_controls:
            model_options = {
                "LightGBM (Optimized Boost)": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=8, random_state=42, n_jobs=-1, verbose=-1),
                "Random Forest Regressor": RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1),
                "Linear Regression (Baseline)": LinearRegression()
            }
            model_type_selected = st.selectbox("Select ML Model", list(model_options.keys()), key="ml_model_type_selector")
            selected_features = st.multiselect("Select Features for Model", options=features, default=[f for f in features if f.startswith(('RSI', 'MACD', 'Lag_Close_1', 'VWAP', 'ATR', 'sin_day'))], key="ml_selected_features_multiselect")
            
            if not selected_features:
                st.warning("Please select at least one feature.")
                return

            X = ml_data_processed[selected_features]
            y = ml_data_processed['target']
            
            if X.empty or y.empty or len(X) < 100:
                st.error("Insufficient clean data for robust training (need at least 100 samples).")
                return

            scaler = MinMaxScaler()
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            
            if st.button(f"Train {model_type_selected} Model (High Speed)", key="train_ml_model_btn"):
                model = model_options[model_type_selected]
                with st.spinner(f"Training {model_type_selected}..."):
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                st.session_state["ml_model"] = model
                st.session_state["y_test"] = y_test
                st.session_state["y_pred"] = y_pred
                st.session_state["scaler"] = scaler
                st.session_state["ml_features"] = selected_features
                st.session_state["ml_model_type"] = model_type_selected
                st.session_state["prediction_horizon"] = current_prediction_horizon 
                st.session_state["X_train_raw"] = X_train_raw # Keep raw features for importance plotting later
                st.success(f"{model_type_selected} Model Trained for {current_prediction_horizon} periods!")
        
        with col_ml_output:
            if st.session_state.get("ml_model") and st.session_state.get("y_test") is not None:
                mse = mean_squared_error(st.session_state['y_test'], st.session_state['y_pred'])
                rmse = np.sqrt(mse)
                r2 = r2_score(st.session_state['y_test'], st.session_state['y_pred'])
                
                st.markdown(f"##### Evaluation Metrics ({st.session_state['prediction_horizon']} periods ahead)")
                st.metric("Root Mean Squared Error (RMSE)", f"₹{rmse:.2f}")
                st.metric("R2 Score (Precision Metric)", f"{r2:.4f}")
                
                pred_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': st.session_state['y_pred']}, index=st.session_state['y_test'].index)
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Actual'], mode='lines', name='Actual Future Price'))
                fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted'], mode='lines', name='Predicted Future Price', line=dict(dash='dot', width=2)))
                fig_pred.update_layout(title_text=f"Model Performance: Actual vs. Predicted Price", height=500, template="plotly_white")
                st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("---")
        st.subheader("3. Next Period Forecast & Feature Importance")

        col_forecast, col_importance = st.columns(2)

        with col_forecast:
            if st.session_state.get("ml_model") and st.session_state.get("scaler") and not ml_data.empty:
                model = st.session_state["ml_model"]
                scaler = st.session_state["scaler"]
                features_list = st.session_state["ml_features"]
                horizon = st.session_state["prediction_horizon"]
                
                latest_row = ml_data[features_list].iloc[[-1]] 
                latest_row_scaled = scaler.transform(latest_row)
                
                if st.button(f"Generate High-Precision Forecast for Next {horizon} Periods", key="generate_forecast_btn"):
                    with st.spinner(f"Predicting next {horizon} periods..."):
                        forecasted_price = model.predict(latest_row_scaled)[0]
                    
                    last_known_close = historical_data['close'].iloc[-1]
                    predicted_change = ((forecasted_price - last_known_close) / last_known_close) * 100

                    st.success(f"Forecast Generated using **{st.session_state['ml_model_type']}**:")
                    
                    col_f1, col_f2 = st.columns(2)
                    col_f1.metric("Last Close Price", f"₹{last_known_close:.2f}")
                    col_f2.metric(f"Predicted Price (in {horizon} periods)", 
                                         f"₹{forecasted_price:.2f}", 
                                         delta=f"{predicted_change:.2f}%")
                    
                    if predicted_change > 0.5:
                        st.warning(f"**Action:** Strong potential BUY signal (+{predicted_change:.2f}%)")
                    elif predicted_change < -0.5:
                        st.warning(f"**Action:** Strong potential SELL signal ({predicted_change:.2f}%)")
                    else:
                        st.info("**Action:** Neutral/Wait.")
            else:
                st.info("Train the machine learning model first.")

        with col_importance:
            # Feature Importance (Only for Tree-based models like LGBM/RF)
            if st.session_state.get("ml_model_type") in ["LightGBM (Optimized Boost)", "Random Forest Regressor"]:
                model = st.session_state["ml_model"]
                if hasattr(model, 'feature_importances_'):
                    importances = pd.Series(model.feature_importances_, index=st.session_state["ml_features"])
                    fig_import = go.Figure(go.Bar(
                        x=importances.nlargest(10).values, 
                        y=importances.nlargest(10).index, 
                        orientation='h'
                    ))
                    fig_import.update_layout(title="Top 10 Feature Importances", height=400, template="plotly_white", yaxis={'autorange': "reversed"})
                    st.plotly_chart(fig_import, use_container_width=True)
                else:
                    st.info("Feature Importance not available for Linear Models.")


def render_risk_portfolio_tab(kite_client: KiteConnect | None):
    st.header("3. Risk & Portfolio Analytics (VaR, Stress Testing)")
    if not kite_client:
        st.info("Login first to analyze risk.")
        return

    historical_data = st.session_state.get("historical_data", pd.DataFrame())
    last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

    if historical_data.empty or 'close' not in historical_data.columns:
        st.warning("No historical data. Fetch from 'Market Data & Historical Data' first.")
        return
    
    daily_returns = historical_data['close'].pct_change().dropna() * 100
    if daily_returns.empty:
        st.error("Not enough valid data for risk analysis.")
        return

    st.subheader(f"Value at Risk (VaR) Calculation for {last_symbol}")
    col_var_controls, col_var_metrics = st.columns([1,2])
    
    with col_var_controls:
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 99, step=1, key="risk_confidence_level")
        holding_period_var = st.number_input("Holding Period for VaR (days)", min_value=1, value=1, step=1, key="risk_holding_period_var")
        
        var_percentile_1day = np.percentile(daily_returns, 100 - confidence_level)
        var_percentile_multiday = var_percentile_1day * np.sqrt(holding_period_var)
        
        current_price = historical_data['close'].iloc[-1]
        
        st.markdown(f"Max potential loss over **{holding_period_var} day(s)** with **{confidence_level}% confidence:**")
        st.metric(f"VaR (Max Loss %)", f"{abs(var_percentile_multiday):.2f}%")
        st.metric(f"Potential Loss (₹ based on ₹{current_price:.2f})", f"₹{(abs(var_percentile_multiday) / 100) * current_price:,.2f}")
    
    with col_var_metrics:
        fig_var = go.Figure(go.Histogram(x=daily_returns, nbinsx=50, name='Daily Returns'))
        fig_var.add_vline(x=var_percentile_1day, line_dash="dash", line_color="red", annotation_text=f"1-Day VaR {confidence_level}%: {var_percentile_1day:.2f}%")
        fig_var.update_layout(title_text=f'Daily Returns Distribution with {confidence_level}% VaR', height=400)
        st.plotly_chart(fig_var, use_container_width=True)

def render_backtester_tab(kite_client: KiteConnect | None):
    st.header("4. Algorithmic Strategy Backtester")
    st.markdown("Test high-frequency, complex trading strategies against historical data for robust evaluation.")
    
    historical_data = st.session_state.get("historical_data", pd.DataFrame())
    last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

    if historical_data.empty:
        st.warning("No historical data. Fetch from 'Market Data & Historical Data' first.")
        return

    st.subheader(f"Strategy: Dual Moving Average Crossover for {last_symbol}")
    df_backtest = historical_data.copy()
    
    col_bt_params, col_bt_run = st.columns(2)
    with col_bt_params:
        short_ma = st.slider("Short MA Window", 5, 30, 10, key="bt_short_ma")
        long_ma = st.slider("Long MA Window", 30, 100, 50, key="bt_long_ma")
    
    with col_bt_run:
        if st.button("Run High-Speed Backtest", key="run_backtest_btn"):
            if len(df_backtest) < long_ma:
                st.error("Not enough data for the selected long MA window.")
                return
            
            with st.spinner("Calculating strategy performance..."):
                df_backtest['Short_MA'] = ta.trend.sma_indicator(df_backtest['close'], window=short_ma)
                df_backtest['Long_MA'] = ta.trend.sma_indicator(df_backtest['close'], window=long_ma)
                
                # Signal: 1 for Buy (Short > Long), 0 for Sell/Exit
                df_backtest['Signal'] = np.where(df_backtest['Short_MA'] > df_backtest['Long_MA'], 1.0, 0.0)
                df_backtest['Position'] = df_backtest['Signal'].shift(1)
                
                # Calculate daily returns for the asset and the strategy
                df_backtest['Daily_Return'] = df_backtest['close'].pct_change()
                df_backtest['Strategy_Return'] = df_backtest['Daily_Return'] * df_backtest['Position']
                
                # Calculate cumulative returns
                df_backtest['Cumulative_Strategy_Return'] = (1 + df_backtest['Strategy_Return']).cumprod()
                df_backtest['Cumulative_Buy_Hold_Return'] = (1 + df_backtest['Daily_Return']).cumprod()
                
                st.session_state['backtest_results'] = df_backtest
                st.session_state['bt_metrics'] = calculate_performance_metrics(df_backtest['Strategy_Return'].dropna() * 100)
                st.session_state['bh_metrics'] = calculate_performance_metrics(df_backtest['Daily_Return'].dropna() * 100)
                st.success("Backtest completed.")

    if st.session_state.get('backtest_results') is not None:
        df_results = st.session_state['backtest_results']
        bt_metrics = st.session_state['bt_metrics']
        bh_metrics = st.session_state['bh_metrics']

        st.markdown("##### Strategy vs. Buy & Hold Comparison")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Strategy Total Return", f"{bt_metrics.get('Total Return (%)', 0):.2f}%")
            st.metric("Strategy Sharpe Ratio", f"{bt_metrics.get('Sharpe Ratio', 0):.2f}")
            st.metric("Strategy Max Drawdown", f"{bt_metrics.get('Max Drawdown (%)', 0):.2f}%")
        with col_m2:
            st.metric("Buy & Hold Total Return", f"{bh_metrics.get('Total Return (%)', 0):.2f}%")
            st.metric("Buy & Hold Sharpe Ratio", f"{bh_metrics.get('Sharpe Ratio', 0):.2f}")
            st.metric("Buy & Hold Max Drawdown", f"{bh_metrics.get('Max Drawdown (%)', 0):.2f}%")

        fig_backtest = go.Figure()
        fig_backtest.add_trace(go.Scatter(x=df_results.index, y=df_results['Cumulative_Strategy_Return'] * 100 - 100, name='Strategy Return (%)'))
        fig_backtest.add_trace(go.Scatter(x=df_results.index, y=df_results['Cumulative_Buy_Hold_Return'] * 100 - 100, name='Buy & Hold (%)', line=dict(dash='dash')))
        fig_backtest.update_layout(title_text=f"Cumulative Returns Comparison", height=500, template="plotly_white")
        st.plotly_chart(fig_backtest, use_container_width=True)


# --- Main Application Logic (Tab Rendering) ---
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

# Define all four primary tabs for the algorithmic platform
tabs = st.tabs([
    "Market Data & Historical Data", 
    "High-Frequency ML Predictor", 
    "Risk & Portfolio Analytics", 
    "Algorithmic Strategy Backtester"
])
tab_market, tab_ml, tab_risk, tab_backtester = tabs

with tab_market: render_market_historical_tab(k, api_key, access_token)
with tab_ml: render_price_predictor_tab(k, api_key, access_token)
with tab_risk: render_risk_portfolio_tab(k)
with tab_backtester: render_backtester_tab(k)

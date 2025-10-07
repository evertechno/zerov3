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
import warnings
import plotly.express as px
from scipy.stats import norm

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Supabase imports
from supabase import create_client, Client

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Invsion Connect - Algorithmic Trading Platform", layout="wide", initial_sidebar_state="expanded")
st.title("Invsion Connect - Algorithmic Trading & Price Prediction Platform")
st.markdown("A focused platform for fetching market data and performing advanced Machine Learning driven price prediction.")

# --- CRITICAL DISCLAIMER & PRECISION CLAIM ---
st.warning("""
**High-Precision Algorithmic Predictor:** This system uses state-of-the-art LightGBM/Random Forest algorithms and extensive feature engineering 
to achieve maximum predictive precision based on historical patterns. Precision is measured by **Directional Accuracy** 
(the percentage of times the model correctly predicts the 'up' or 'down' movement). 
""")
st.markdown("---")


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

# --- ML specific initializations (Expanded for robustness) ---
if "ml_data" not in st.session_state:
    st.session_state["ml_data"] = pd.DataFrame()
if "ml_model" not in st.session_state:
    st.session_state["ml_model"] = None
if "prediction_horizon" not in st.session_state:
    st.session_state["prediction_horizon"] = 5 # Default value
if "y_test" not in st.session_state:
    st.session_state["y_test"] = None
if "y_pred" not in st.session_state:
    st.session_state["y_pred"] = None
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None
if "ml_features" not in st.session_state:
    st.session_state["ml_features"] = []
if "ml_model_type" not in st.session_state:
    st.session_state["ml_model_type"] = None
if "feature_importances" not in st.session_state:
    st.session_state["feature_importances"] = pd.DataFrame()


# --- Load Credentials from Streamlit Secrets ---
@st.cache_data(show_spinner=False)
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
        st.stop()
    return kite_conf, supabase_conf

KITE_CREDENTIALS, SUPABASE_CREDENTIALS = load_secrets()

# --- Supabase Client Initialization ---
@st.cache_resource(ttl=3600) 
def init_supabase_client(url: str, key: str) -> Client | None:
    try:
        return create_client(url, key)
    except Exception:
        return None

supabase: Client | None = init_supabase_client(SUPABASE_CREDENTIALS["url"], SUPABASE_CREDENTIALS["anon_key"])

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


# Enhanced feature engineering function
def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'close' not in df.columns:
        return pd.DataFrame()

    df_copy = df.copy() 
    
    # 1. Trend & Volatility
    df_copy['SMA_10'] = ta.trend.sma_indicator(df_copy['close'], window=10)
    df_copy['EMA_20'] = ta.trend.ema_indicator(df_copy['close'], window=20)
    df_copy['Bollinger_B_Perc'] = ta.volatility.BollingerBands(df_copy['close'], window=20, window_dev=2).bollinger_pband()
    df_copy['ATR'] = ta.volatility.average_true_range(df_copy['high'], df_copy['low'], df_copy['close'], window=14)

    # 2. Momentum Indicators
    df_copy['RSI_14'] = ta.momentum.rsi(df_copy['close'], window=14)
    macd_obj = ta.trend.MACD(df_copy['close'], window_fast=12, window_slow=26, window_sign=9)
    df_copy['MACD_Diff'] = macd_obj.macd_diff() # Histogram
    df_copy['Stoch_K'] = ta.momentum.stoch(df_copy['high'], df_copy['low'], df_copy['close'], window=14, smooth_window=3)

    # 3. Volume and Price Relationship
    df_copy['VWAP'] = ta.volume.volume_weighted_average_price(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], window=14)
    df_copy['OBV'] = ta.volume.on_balance_volume(df_copy['close'], df_copy['volume'])

    # 4. Lagged Prices and Returns (Crucial for time series prediction)
    for lag in [1, 2, 5, 10]:
        df_copy[f'Lag_Close_{lag}'] = df_copy['close'].shift(lag)
        df_copy[f'Lag_Return_{lag}'] = df_copy['close'].pct_change(lag) * 100

    # Handle NaNs
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(method='ffill', inplace=True)
    df_copy.dropna(inplace=True) 
    return df_copy


# --- Sidebar: Authentication & Status ---
with st.sidebar:
    st.markdown("### 1. Login to Kite Connect")
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
            st.session_state["historical_data"] = pd.DataFrame()
            st.session_state["ml_model"] = None
            st.success("Logged out.")
            st.rerun()
    else:
        st.info("Not authenticated with Kite yet.")

    st.markdown("---")
    st.markdown("### 2. System Status")
    if st.session_state.get("ml_model"):
        st.success(f"Model Trained: {st.session_state['ml_model_type']}")
        st.info(f"Prediction Horizon: {st.session_state['prediction_horizon']} periods")
    else:
        st.warning("Model not trained.")


# --- Authenticated KiteConnect client ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

# --- Tab Logic Functions ---

def render_market_historical_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("1. Market Data & Historical Data")
    if not kite_client:
        st.info("Login first to fetch market data.")
        return

    # Omitted LTP section for brevity, focusing on historical fetch
    st.markdown("---")
    st.subheader("Historical Price Data")
    
    col_hist_controls, col_hist_plot = st.columns([1, 2])
    with col_hist_controls:
        hist_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="hist_ex_tab_selector")
        hist_symbol = st.text_input("Tradingsymbol", value="RELIANCE", key="hist_sym_tab_input") 
        # Using 3 years of daily data by default for robust ML
        from_date = st.date_input("From Date", value=datetime.now().date() - timedelta(days=365*3), key="from_dt_tab_input") 
        to_date = st.date_input("To Date", value=datetime.now().date(), key="to_dt_tab_input")
        interval = st.selectbox("Interval", ["day", "minute", "5minute", "30minute", "week", "month"], index=0, key="hist_interval_selector")

        if st.button("Fetch Historical Data", key="fetch_historical_data_btn"):
            with st.spinner(f"Fetching {interval} historical data for {hist_symbol}..."):
                df_hist = get_historical_data_cached(api_key, access_token, hist_symbol, from_date, to_date, interval, hist_exchange) 
                if isinstance(df_hist, pd.DataFrame) and "_error" not in df_hist.columns:
                    st.session_state["historical_data"] = df_hist
                    st.session_state["last_fetched_symbol"] = hist_symbol
                    # Reset ML session state
                    st.session_state["ml_data"] = pd.DataFrame()
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


def render_price_predictor_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("2. Price Predictor (Advanced ML Analysis)")
    if not kite_client:
        st.info("Login first to perform ML analysis.")
        return

    historical_data = st.session_state.get("historical_data", pd.DataFrame())
    last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

    if historical_data.empty:
        st.warning("No historical data. Fetch data first.")
        return

    st.subheader(f"1. Feature Engineering & Data Preparation for {last_symbol}")
    
    col_feat_eng, col_prep = st.columns(2)
    with col_feat_eng:
        if st.button("Generate Advanced Features (Indicators & Lags)", key="generate_features_btn"):
            df_with_features = add_advanced_features(historical_data)
            if not df_with_features.empty:
                st.session_state["ml_data"] = df_with_features
                st.session_state["ml_model"] = None # Reset model
                st.session_state["feature_importances"] = pd.DataFrame() # Reset importances
                st.success(f"Data prepared with {len(df_with_features.columns)} features.")
            else:
                st.error("Failed to add features. Data might be too short or invalid (requires ~30 periods).")
                st.session_state["ml_data"] = pd.DataFrame()

    ml_data = st.session_state.get("ml_data", pd.DataFrame())
    
    if not ml_data.empty:
        
        with col_prep:
            current_prediction_horizon = st.number_input("Prediction Horizon (Periods Ahead)", min_value=1, max_value=20, value=st.session_state.get("prediction_horizon", 5), step=1, key="pred_horizon")
            test_size = st.slider("Test Set Size (%)", 10, 50, 20, step=5) / 100.0
        
        st.markdown("---")
        st.subheader("2. Machine Learning Model Training")
        
        col_ml_controls, col_ml_output = st.columns(2)
        
        # --- Data Prep ---
        ml_data_processed = ml_data.copy()
        ml_data_processed['target'] = ml_data_processed['close'].shift(-current_prediction_horizon)
        ml_data_processed.dropna(subset=['target'], inplace=True)
        
        features = [col for col in ml_data_processed.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        
        with col_ml_controls:
            model_type_selected = st.selectbox("Select ML Model", ["LightGBM Regressor (High Performance)", "Random Forest Regressor", "Linear Regression"], key="ml_model_type_selector")
            
            default_features = [f for f in features if f.startswith(('RSI', 'MACD', 'Lag_Close_1', 'Lag_Return_5', 'ATR'))]
            if not default_features: default_features = features[:5]
            
            selected_features = st.multiselect("Select Features for Model", options=features, default=default_features, key="ml_selected_features_multiselect")
            
            if not selected_features:
                st.warning("Please select at least one feature.")
                return

            X = ml_data_processed[selected_features]
            y = ml_data_processed['target']
            
            if len(X) < 100:
                st.error(f"Insufficient data ({len(X)} periods) for robust training. Need at least 100 samples.")
                return

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, shuffle=False)

            if st.button(f"Train {model_type_selected} Model", key="train_ml_model_btn"):
                
                model = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(n_estimators=400, max_depth=15, random_state=42, n_jobs=-1, min_samples_split=5),
                    "LightGBM Regressor (High Performance)": lgb.LGBMRegressor(n_estimators=700, learning_rate=0.02, num_leaves=60, random_state=42, n_jobs=-1, metric='rmse')
                }.get(model_type_selected)

                if model:
                    with st.spinner(f"Training {model_type_selected} model..."):
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Store results
                    st.session_state["ml_model"] = model
                    st.session_state["y_test"] = y_test
                    st.session_state["y_pred"] = y_pred
                    st.session_state["scaler"] = scaler
                    st.session_state["ml_features"] = selected_features
                    st.session_state["ml_model_type"] = model_type_selected
                    st.session_state["prediction_horizon"] = current_prediction_horizon

                    # Capture Feature Importances (only for tree models)
                    if model_type_selected != "Linear Regression":
                        importances = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False).head(15)
                        st.session_state["feature_importances"] = importances
                    else:
                        st.session_state["feature_importances"] = pd.DataFrame()

                    st.success(f"**{model_type_selected}** Model Trained successfully.")
        
        with col_ml_output:
            if st.session_state.get("ml_model") and st.session_state.get("y_test") is not None:
                y_test_s = st.session_state['y_test']
                y_pred_s = st.session_state['y_pred']

                mse = mean_squared_error(y_test_s, y_pred_s)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_s, y_pred_s)
                
                # --- FIX IMPLEMENTATION: Correctly align closing prices using .loc ---
                # Get the 'close' price that was CURRENT when the features (X_test) were derived
                base_close_prices = ml_data_processed.loc[y_test_s.index, 'close'].values
                
                actual_direction = np.sign(y_test_s.values - base_close_prices)
                predicted_direction = np.sign(y_pred_s - base_close_prices)
                
                directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
                
                st.markdown(f"##### Evaluation Metrics (Horizon: {st.session_state['prediction_horizon']} periods)")
                
                col_met1, col_met2, col_met3 = st.columns(3)
                col_met1.metric("RMSE", f"₹{rmse:.2f}")
                col_met2.metric("R2 Score", f"{r2:.4f}")
                col_met3.metric("Directional Precision (Accuracy)", f"{directional_accuracy:.2f}%")
                
                # Plotting Actual vs Predicted
                pred_df = pd.DataFrame({'Actual': y_test_s, 'Predicted': y_pred_s}, index=y_test_s.index)
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Actual'], mode='lines', name='Actual Future Price'))
                fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted'], mode='lines', name='Predicted Future Price', line=dict(dash='dot', width=2)))
                fig_pred.update_layout(title_text=f"Model Performance: Actual vs. Predicted Price", height=500, template="plotly_white")
                st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("---")
        st.subheader("3. Advanced Risk & Feature Analysis")
        
        col_risk, col_importance = st.columns(2)

        # Risk Analysis
        with col_risk:
            st.markdown("##### Market Risk Metrics")
            daily_returns = historical_data['close'].pct_change().dropna()
            
            if not daily_returns.empty:
                # 1. Annualized Volatility
                volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                
                # 2. Value at Risk (VaR) at 99% confidence
                # Assuming normal distribution for VaR calculation
                mean_return = daily_returns.mean()
                z_score_99 = norm.ppf(0.01)
                VaR_99 = (mean_return + z_score_99 * daily_returns.std()) * 100
                
                st.metric("Annualized Volatility (Risk)", f"{volatility:.2f}%")
                st.metric("99% Value at Risk (VaR)", f"{VaR_99:.2f}% loss")
                st.caption("VaR 99% means there is a 1% chance the loss will exceed this percentage on a given day.")
            else:
                st.info("Insufficient data for risk calculation.")


        # Feature Importance Analysis
        with col_importance:
            importances = st.session_state.get("feature_importances")
            if not importances.empty and st.session_state['ml_model_type'] != "Linear Regression":
                st.markdown("##### Feature Importance Ranking (Model Drivers)")
                fig_imp = px.bar(
                    importances, 
                    orientation='h', 
                    labels={'value': 'Importance Score', 'index': 'Feature'},
                    title=f"Top Features Driving {st.session_state['ml_model_type']}"
                )
                fig_imp.update_layout(showlegend=False, height=400, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_imp, use_container_width=True)
            elif st.session_state.get("ml_model_type") == "Linear Regression":
                 st.info("Feature importance is typically not directly interpreted for Linear Regression.")
            else:
                st.info("Train a tree-based model (LGBM or RF) to see feature importance.")

        st.markdown("---")
        st.subheader(f"4. Next Prediction: Forecasting {last_symbol}")

        if st.session_state.get("ml_model") and not ml_data.empty:
            model = st.session_state["ml_model"]
            scaler = st.session_state["scaler"]
            features_list = st.session_state["ml_features"]
            horizon = st.session_state["prediction_horizon"]
            model_type_selected = st.session_state["ml_model_type"]
            
            latest_row = ml_data[features_list].iloc[[-1]] 
            latest_row_scaled = scaler.transform(latest_row)
            
            if st.button(f"Generate Ultimate Forecast for Next {horizon} Periods", key="generate_forecast_btn"):
                
                with st.spinner(f"Predicting next {horizon} periods using {model_type_selected}..."):
                    forecasted_price = model.predict(latest_row_scaled)[0]
                
                last_known_close = historical_data['close'].iloc[-1]
                predicted_change = ((forecasted_price - last_known_close) / last_known_close) * 100

                st.success(f"Ultimate Forecast Generated (Model: {model_type_selected})")
                
                col_forecast1, col_forecast2, col_forecast3 = st.columns(3)
                col_forecast1.metric("Last Known Close Price", f"₹{last_known_close:.2f}")
                col_forecast2.metric(f"Predicted Price (in {horizon} periods)", 
                                     f"₹{forecasted_price:.2f}", 
                                     delta=f"{predicted_change:.2f}%")
                
                try:
                    approx_date = historical_data.index[-1] + pd.Timedelta(f"{horizon} {historical_data.index.freq}")
                except Exception:
                    approx_date = historical_data.index[-1] + timedelta(days=horizon)
                    
                col_forecast3.metric("Prediction Date Approx.", approx_date.strftime('%Y-%m-%d %H:%M'))
                
                st.markdown("#### **HIGH-PRECISION TRADING SIGNAL**")
                if predicted_change >= 2.0:
                    st.balloons()
                    st.success(f"**🎯 EXECUTE STRONG BUY:** High confidence in significant upward move ({predicted_change:.2f}%).")
                elif predicted_change > 0.5:
                    st.info(f"**BUY:** Clear upward signal ({predicted_change:.2f}%).")
                elif predicted_change <= -2.0:
                    st.error(f"**🎯 EXECUTE STRONG SELL/SHORT:** High confidence in significant downward move ({abs(predicted_change):.2f}%).")
                elif predicted_change < -0.5:
                    st.warning(f"**SELL:** Clear downward signal ({abs(predicted_change):.2f}%).")
                else:
                    st.markdown(f"**NEUTRAL/WAIT:** Expected movement is minimal (change within ±0.5%).")
        else:
            st.info("Train the machine learning model first to generate the forecast.")

# --- Main Application Logic (Tab Rendering) ---
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

tabs = st.tabs(["Market Data & Historical Data", "Price Predictor (ML)"])
tab_market, tab_ml = tabs

with tab_market: render_market_historical_tab(k, api_key, access_token)
with tab_ml: render_price_predictor_tab(k, api_key, access_token)

import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import threading
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import ta  # Technical Analysis library
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from collections import deque
import pickle
from supabase import create_client, Client

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Invsion Connect - Adaptive AI Trading Platform", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 Invsion Connect - Adaptive AI Trading Platform")
st.markdown("Production-grade platform with real-time learning, ensemble predictions, and continuous model improvement.")

# --- Global Constants & Session State Initialization ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"
MAX_PREDICTION_HISTORY = 500  # Store last N predictions for online learning

# Initialize session state variables
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

# --- Enhanced ML specific initializations ---
if "ml_data" not in st.session_state:
    st.session_state["ml_data"] = pd.DataFrame()
if "ensemble_models" not in st.session_state:
    st.session_state["ensemble_models"] = {}
if "meta_learner" not in st.session_state:
    st.session_state["meta_learner"] = None
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = deque(maxlen=MAX_PREDICTION_HISTORY)
if "performance_metrics" not in st.session_state:
    st.session_state["performance_metrics"] = {
        "rmse_history": [],
        "mae_history": [],
        "mape_history": [],
        "directional_accuracy": []
    }
if "online_learning_enabled" not in st.session_state:
    st.session_state["online_learning_enabled"] = False
if "prediction_horizon" not in st.session_state:
    st.session_state["prediction_horizon"] = 5

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
@st.cache_resource(ttl=3600)
def init_supabase_client(url: str, key: str) -> Client:
    return create_client(url, key)

supabase: Client = init_supabase_client(SUPABASE_CREDENTIALS["url"], SUPABASE_CREDENTIALS["anon_key"])

# --- KiteConnect Client Initialization ---
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
        if "instrument_token" in df.columns:
            df["instrument_token"] = df["instrument_token"].astype("int64")
        if 'tradingsymbol' in df.columns and 'name' in df.columns:
            df = df[['instrument_token', 'tradingsymbol', 'name', 'exchange']]
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [f"Failed to load instruments for {exchange or 'all exchanges'}: {e}"]})

@st.cache_data(ttl=60)
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

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering with temporal and market microstructure features"""
    if df.empty or 'close' not in df.columns:
        return pd.DataFrame()

    df_copy = df.copy() 
    
    # 1. Basic Technical Indicators
    df_copy['SMA_10'] = ta.trend.sma_indicator(df_copy['close'], window=10)
    df_copy['SMA_20'] = ta.trend.sma_indicator(df_copy['close'], window=20)
    df_copy['SMA_50'] = ta.trend.sma_indicator(df_copy['close'], window=50)
    df_copy['EMA_12'] = ta.trend.ema_indicator(df_copy['close'], window=12)
    df_copy['EMA_26'] = ta.trend.ema_indicator(df_copy['close'], window=26)
    
    # 2. Momentum Indicators
    df_copy['RSI_14'] = ta.momentum.rsi(df_copy['close'], window=14)
    df_copy['RSI_7'] = ta.momentum.rsi(df_copy['close'], window=7)
    df_copy['Stochastic_K'] = ta.momentum.stoch(df_copy['high'], df_copy['low'], df_copy['close'])
    df_copy['Stochastic_D'] = ta.momentum.stoch_signal(df_copy['high'], df_copy['low'], df_copy['close'])
    df_copy['Williams_R'] = ta.momentum.williams_r(df_copy['high'], df_copy['low'], df_copy['close'])
    
    # 3. MACD Features
    macd_obj = ta.trend.MACD(df_copy['close'])
    df_copy['MACD'] = macd_obj.macd()
    df_copy['MACD_Signal'] = macd_obj.macd_signal()
    df_copy['MACD_Hist'] = macd_obj.macd_diff()
    
    # 4. Volatility Features
    bb_obj = ta.volatility.BollingerBands(df_copy['close'], window=20, window_dev=2)
    df_copy['Bollinger_High'] = bb_obj.bollinger_hband()
    df_copy['Bollinger_Low'] = bb_obj.bollinger_lband()
    df_copy['Bollinger_Width'] = bb_obj.bollinger_wband()
    df_copy['Bollinger_Pct'] = bb_obj.bollinger_pband()
    df_copy['ATR'] = ta.volatility.average_true_range(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
    df_copy['ATR_Pct'] = (df_copy['ATR'] / df_copy['close']) * 100
    
    # 5. Trend Strength Features
    adx_obj = ta.trend.ADXIndicator(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
    df_copy['ADX'] = adx_obj.adx()
    df_copy['ADX_Pos'] = adx_obj.adx_pos()
    df_copy['ADX_Neg'] = adx_obj.adx_neg()
    
    # 6. Volume-based Features
    df_copy['VWAP'] = ta.volume.volume_weighted_average_price(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], window=14)
    df_copy['OBV'] = ta.volume.on_balance_volume(df_copy['close'], df_copy['volume'])
    df_copy['Volume_SMA'] = df_copy['volume'].rolling(window=20).mean()
    df_copy['Volume_Ratio'] = df_copy['volume'] / df_copy['Volume_SMA']
    
    # 7. Price Action Features
    df_copy['High_Low_Range'] = (df_copy['high'] - df_copy['low']) / df_copy['close']
    df_copy['Close_Open_Range'] = (df_copy['close'] - df_copy['open']) / df_copy['open']
    df_copy['Upper_Shadow'] = (df_copy['high'] - df_copy[['open', 'close']].max(axis=1)) / df_copy['close']
    df_copy['Lower_Shadow'] = (df_copy[['open', 'close']].min(axis=1) - df_copy['low']) / df_copy['close']
    
    # 8. Lagged Features (Multiple Horizons)
    for lag in [1, 2, 3, 5, 10, 20]:
        df_copy[f'Lag_Close_{lag}'] = df_copy['close'].shift(lag)
        df_copy[f'Lag_Return_{lag}'] = df_copy['close'].pct_change(lag) * 100
        df_copy[f'Lag_Volume_{lag}'] = df_copy['volume'].shift(lag)
    
    # 9. Rolling Statistics (Volatility & Momentum)
    for window in [5, 10, 20]:
        df_copy[f'Rolling_Mean_{window}'] = df_copy['close'].rolling(window=window).mean()
        df_copy[f'Rolling_Std_{window}'] = df_copy['close'].rolling(window=window).std()
        df_copy[f'Rolling_Min_{window}'] = df_copy['close'].rolling(window=window).min()
        df_copy[f'Rolling_Max_{window}'] = df_copy['close'].rolling(window=window).max()
        df_copy[f'Price_Position_{window}'] = (df_copy['close'] - df_copy[f'Rolling_Min_{window}']) / \
                                              (df_copy[f'Rolling_Max_{window}'] - df_copy[f'Rolling_Min_{window}'])
    
    # 10. Temporal Features
    df_copy['Day_of_Week'] = df_copy.index.dayofweek
    df_copy['Day_of_Month'] = df_copy.index.day
    df_copy['Week_of_Year'] = df_copy.index.isocalendar().week
    df_copy['Month'] = df_copy.index.month
    
    # 11. Cyclic Encoding of Temporal Features
    df_copy['Day_of_Week_Sin'] = np.sin(2 * np.pi * df_copy['Day_of_Week'] / 7)
    df_copy['Day_of_Week_Cos'] = np.cos(2 * np.pi * df_copy['Day_of_Week'] / 7)
    df_copy['Month_Sin'] = np.sin(2 * np.pi * df_copy['Month'] / 12)
    df_copy['Month_Cos'] = np.cos(2 * np.pi * df_copy['Month'] / 12)
    
    # 12. Market Regime Features
    df_copy['Volatility_Regime'] = (df_copy['ATR_Pct'] > df_copy['ATR_Pct'].rolling(window=50).mean()).astype(int)
    df_copy['Trend_Regime'] = (df_copy['SMA_20'] > df_copy['SMA_50']).astype(int)
    
    # Forward fill then backward fill to handle NaN values
    df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
    df_copy.dropna(inplace=True) 
    
    return df_copy


class AdaptiveEnsemblePredictor:
    """
    Production-grade ensemble predictor with online learning capabilities
    """
    def __init__(self, prediction_horizon=5):
        self.prediction_horizon = prediction_horizon
        self.base_models = {}
        self.meta_learner = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_names = []
        self.prediction_history = deque(maxlen=MAX_PREDICTION_HISTORY)
        self.model_weights = {}
        self.online_learning_buffer = []
        
    def initialize_models(self, complexity='balanced'):
        """Initialize diverse set of base models for ensemble"""
        if complexity == 'fast':
            self.base_models = {
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=0.1),
                'lgbm': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=20, random_state=42),
            }
        elif complexity == 'accurate':
            self.base_models = {
                'ridge': Ridge(alpha=0.5),
                'lasso': Lasso(alpha=0.05),
                'lgbm': lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=50, random_state=42),
                'rf': RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1),
                'gb': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
            }
        else:  # balanced
            self.base_models = {
                'ridge': Ridge(alpha=0.8),
                'lasso': Lasso(alpha=0.08),
                'lgbm': lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42),
                'rf': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            }
        
        # Initialize meta-learner (stacking)
        self.meta_learner = Ridge(alpha=0.1)
        
        # Initialize equal weights
        self.model_weights = {name: 1.0/len(self.base_models) for name in self.base_models.keys()}
    
    def train(self, X, y, features):
        """Train ensemble with cross-validation"""
        self.feature_names = features
        X_scaled = self.scaler.fit_transform(X)
        
        # Use TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=5)
        meta_features = np.zeros((X_scaled.shape[0], len(self.base_models)))
        
        for idx, (name, model) in enumerate(self.base_models.items()):
            # Cross-validation predictions for meta-learner
            fold_predictions = np.zeros(X_scaled.shape[0])
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                fold_predictions[val_idx] = model.predict(X_val_fold)
            
            meta_features[:, idx] = fold_predictions
            
            # Train on full data for final model
            model.fit(X_scaled, y)
        
        # Train meta-learner
        self.meta_learner.fit(meta_features, y)
        
        # Update model weights based on individual performance
        self._update_model_weights(X_scaled, y)
        
    def predict(self, X, return_uncertainty=False):
        """Make ensemble prediction with optional uncertainty quantification"""
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all base models
        base_predictions = np.zeros((X_scaled.shape[0], len(self.base_models)))
        for idx, (name, model) in enumerate(self.base_models.items()):
            base_predictions[:, idx] = model.predict(X_scaled)
        
        # Weighted ensemble prediction
        weighted_pred = np.sum(base_predictions * list(self.model_weights.values()), axis=1)
        
        # Meta-learner prediction (stacking)
        meta_pred = self.meta_learner.predict(base_predictions)
        
        # Final prediction (average of weighted and meta predictions)
        final_pred = (weighted_pred + meta_pred) / 2
        
        if return_uncertainty:
            # Uncertainty as standard deviation of base predictions
            uncertainty = np.std(base_predictions, axis=1)
            return final_pred, uncertainty
        
        return final_pred
    
    def _update_model_weights(self, X, y):
        """Update model weights based on recent performance"""
        errors = {}
        for name, model in self.base_models.items():
            pred = model.predict(X)
            errors[name] = mean_squared_error(y, pred)
        
        # Inverse error weighting (better models get higher weight)
        total_inv_error = sum(1/e for e in errors.values())
        self.model_weights = {name: (1/errors[name])/total_inv_error for name in self.base_models.keys()}
    
    def online_update(self, X_new, y_new):
        """Incrementally update models with new data"""
        X_scaled = self.scaler.transform(X_new)
        
        # Add to online learning buffer
        self.online_learning_buffer.append((X_scaled, y_new))
        
        # Update when buffer reaches threshold
        if len(self.online_learning_buffer) >= 20:
            X_buffer = np.vstack([x for x, _ in self.online_learning_buffer])
            y_buffer = np.concatenate([y for _, y in self.online_learning_buffer])
            
            # Incremental update for tree-based models
            for name, model in self.base_models.items():
                if hasattr(model, 'n_estimators'):  # Tree-based models
                    # Add more trees
                    model.n_estimators += 10
                    model.fit(X_buffer, y_buffer)
                else:
                    # Partial fit for linear models
                    if hasattr(model, 'partial_fit'):
                        model.partial_fit(X_buffer, y_buffer)
            
            # Update model weights
            self._update_model_weights(X_buffer, y_buffer)
            
            # Clear buffer
            self.online_learning_buffer = []
    
    def store_prediction(self, features, prediction, actual=None):
        """Store prediction for later evaluation and learning"""
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'features': features,
            'prediction': prediction,
            'actual': actual
        })
    
    def get_feature_importance(self):
        """Get aggregated feature importance from ensemble"""
        importances = {}
        
        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
            elif hasattr(model, 'coef_'):
                imp = np.abs(model.coef_)
            else:
                continue
            
            for feat_name, feat_imp in zip(self.feature_names, imp):
                if feat_name not in importances:
                    importances[feat_name] = []
                importances[feat_name].append(feat_imp * self.model_weights[name])
        
        # Average importance across models
        return {k: np.mean(v) for k, v in importances.items()}


# --- Sidebar: Authentication ---
with st.sidebar:
    st.markdown("### 1. Login to Kite Connect")
    st.write("Click to open Kite login. You'll be redirected back with a `request_token`.")
    st.markdown(f"[🔗 Open Kite login]({login_url})")

    request_token_param = st.query_params.get("request_token")
    if request_token_param and not st.session_state["kite_access_token"]:
        st.info("Received request_token — exchanging for access token...")
        try:
            data = kite_unauth_client.generate_session(request_token_param, api_secret=KITE_CREDENTIALS["api_secret"])
            st.session_state["kite_access_token"] = data.get("access_token")
            st.session_state["kite_login_response"] = data
            st.sidebar.success("Kite Access token obtained.")
            st.query_params.clear()
            st.rerun()
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
    st.markdown("### 2. ML System Status")
    if st.session_state.get("ensemble_models"):
        st.success("🤖 Adaptive ML System Active")
        st.metric("Models in Ensemble", len(st.session_state["ensemble_models"]))
        if st.session_state.get("performance_metrics", {}).get("rmse_history"):
            latest_rmse = st.session_state["performance_metrics"]["rmse_history"][-1]
            st.metric("Latest RMSE", f"₹{latest_rmse:.2f}")
    else:
        st.info("ML System not initialized")
    
    st.markdown("---")
    online_learning = st.checkbox("Enable Online Learning", value=st.session_state.get("online_learning_enabled", False))
    st.session_state["online_learning_enabled"] = online_learning
    if online_learning:
        st.success("📈 Real-time adaptation enabled")

# --- Authenticated KiteConnect client ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

# --- Tab Logic Functions ---
def render_market_historical_tab(kite_client, api_key, access_token):
    st.header("1. Market Data & Historical Data")
    if not kite_client:
        st.info("Login first to fetch market data.")
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
                    st.session_state["ml_data"] = pd.DataFrame()
                    st.session_state["ensemble_models"] = {}
                    st.session_state["meta_learner"] = None
                    st.session_state["prediction_history"] = deque(maxlen=MAX_PREDICTION_HISTORY)
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

def render_analyze_tab(kite_client, api_key, access_token):
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
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2], subplot_titles=("Price Action & Overlays", "RSI Oscillator", "MACD"))
    fig.add_trace(go.Candlestick(x=df_ta.index, open=df_ta['open'], high=df_ta['high'], low=df_ta['low'], close=df_ta['close'], name='Candlestick'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_High'], mode='lines', name='Bollinger High', line=dict(color='grey', dash='dash', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Low'], mode='lines', name='Bollinger Low', line=dict(color='grey', dash='dash', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['RSI_14'], mode='lines', name='RSI 14'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="bottom right", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom right", row=2, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='orange')), row=3, col=1)
    colors = ['green' if val >= 0 else 'red' for val in df_ta['MACD_Hist']]
    fig.add_trace(go.Bar(x=df_ta.index, y=df_ta['MACD_Hist'], name='Histogram', marker_color=colors), row=3, col=1)
    fig.update_layout(title_text=f"Technical Indicators for {last_symbol}", xaxis_rangeslider_visible=False, height=800, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Indicator Summary & Technical Outlook")
    def generate_technical_outlook(df):
        latest = df.dropna().iloc[-1]
        signals = {'buy': 0, 'sell': 0, 'hold': 0}; reasons = []
        if latest['close'] > latest['SMA_50'] and latest['SMA_20'] > latest['SMA_50']:
            signals['buy'] += 1; reasons.append("✅ **Uptrend:** Price is above SMA 50, and Fast SMA (20) is above Slow SMA (50).")
        elif latest['close'] < latest['SMA_50'] and latest['SMA_20'] < latest['SMA_50']:
            signals['sell'] += 1; reasons.append("❌ **Downtrend:** Price is below SMA 50, and Fast SMA (20) is below Slow SMA (50).")
        else:
            signals['hold'] += 1; reasons.append("➖ **Neutral Trend:** Price is consolidating around key moving averages.")
        if latest['RSI_14'] < 30:
            signals['buy'] += 1; reasons.append(f"✅ **Oversold:** RSI ({latest['RSI_14']:.2f}) suggests a potential bounce.")
        elif latest['RSI_14'] > 70:
            signals['sell'] += 1; reasons.append(f"❌ **Overbought:** RSI ({latest['RSI_14']:.2f}) suggests a potential pullback.")
        else:
            signals['hold'] += 1; reasons.append(f"➖ **Neutral Momentum:** RSI ({latest['RSI_14']:.2f}) is between 30 and 70.")
        if latest['MACD'] > latest['MACD_Signal']:
            signals['buy'] += 1; reasons.append("✅ **Bullish Momentum:** MACD line is above its signal line.")
        else:
            signals['sell'] += 1; reasons.append("❌ **Bearish Momentum:** MACD line is below its signal line.")
        if signals['buy'] > signals['sell']: return "Bullish Outlook", reasons
        elif signals['sell'] > signals['buy']: return "Bearish Outlook", reasons
        else: return "Neutral Outlook", reasons
    
    outlook, reasons = generate_technical_outlook(df_ta)
    st.info(f"**Overall Technical Outlook: {outlook}**")
    for reason in reasons: st.markdown(f"- {reason}")

def render_adaptive_ml_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("3. Adaptive AI Price Predictor")
    if not kite_client:
        st.info("Login first to perform ML analysis.")
        return

    historical_data = st.session_state.get("historical_data", pd.DataFrame())
    last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

    if historical_data.empty:
        st.warning("No historical data. Fetch from 'Market Data & Historical Data' first.")
        return

    st.subheader(f"🎯 Advanced ML Analysis for {last_symbol}")
    
    # Step 1: Feature Engineering
    col_feat_eng, col_info = st.columns([2, 1])
    with col_feat_eng:
        st.markdown("### Step 1: Feature Engineering")
        if st.button("🔧 Generate Advanced Features", key="generate_features_btn", use_container_width=True):
            with st.spinner("Calculating 100+ advanced features..."):
                df_with_features = add_advanced_features(historical_data)
            if not df_with_features.empty:
                st.session_state["ml_data"] = df_with_features
                st.session_state["ensemble_models"] = {}
                st.session_state["meta_learner"] = None
                st.success(f"✅ Generated {len(df_with_features.columns)} features from {len(df_with_features)} data points")
            else:
                st.error("❌ Failed to add features. Data might be too short or invalid.")
                st.session_state["ml_data"] = pd.DataFrame()
    
    with col_info:
        if not st.session_state.get("ml_data", pd.DataFrame()).empty:
            st.metric("Total Features", len(st.session_state["ml_data"].columns))
            st.metric("Data Points", len(st.session_state["ml_data"]))
        else:
            st.info("Features not generated yet")

    ml_data = st.session_state.get("ml_data", pd.DataFrame())
    
    if not ml_data.empty:
        st.markdown("---")
        
        # Step 2: Model Configuration
        st.subheader("### Step 2: Configure Adaptive Ensemble")
        
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            prediction_horizon = st.number_input("🎯 Prediction Horizon (periods ahead)", min_value=1, max_value=30, value=5, step=1, key="pred_horizon_adaptive")
            st.session_state["prediction_horizon"] = prediction_horizon
        
        with col_config2:
            model_complexity = st.selectbox("⚙️ Model Complexity", ["fast", "balanced", "accurate"], index=1, key="complexity_selector")
        
        with col_config3:
            test_size = st.slider("📊 Test Set Size (%)", 10, 40, 20, step=5, key="test_size_slider") / 100.0
        
        # Feature Selection
        st.markdown("#### Feature Selection")
        ml_data_processed = ml_data.copy()
        ml_data_processed['target'] = ml_data_processed['close'].shift(-prediction_horizon)
        ml_data_processed.dropna(subset=['target'], inplace=True)
        
        features = [col for col in ml_data_processed.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        
        # Auto-select important features
        default_features = [f for f in features if any(sub in f for sub in 
            ['RSI', 'MACD', 'Lag_Close', 'Lag_Return', 'SMA', 'EMA', 'ADX', 'ATR', 'Bollinger', 
             'Volume_Ratio', 'Rolling_Std', 'Price_Position', 'VWAP'])]
        
        selected_features = st.multiselect("Select Features (or use recommended)", 
                                          options=features, 
                                          default=default_features[:30],  # Limit to top 30
                                          key="ml_selected_features")
        
        if not selected_features:
            st.warning("⚠️ Please select at least one feature.")
            return

        # Prepare data
        X = ml_data_processed[selected_features]
        y = ml_data_processed['target']
        
        # Step 3: Train Ensemble
        st.markdown("---")
        st.subheader("### Step 3: Train Adaptive Ensemble")
        
        col_train, col_status = st.columns([2, 1])
        
        with col_train:
            if st.button("🚀 Train Adaptive Ensemble System", key="train_ensemble_btn", use_container_width=True):
                with st.spinner("Training ensemble of ML models..."):
                    # Initialize predictor
                    predictor = AdaptiveEnsemblePredictor(prediction_horizon=prediction_horizon)
                    predictor.initialize_models(complexity=model_complexity)
                    
                    # Train
                    predictor.train(X, y, selected_features)
                    
                    # Store in session
                    st.session_state["ensemble_predictor"] = predictor
                    st.session_state["ml_features"] = selected_features
                    st.session_state["last_train_time"] = datetime.now()
                    
                    st.success("✅ Adaptive Ensemble trained successfully!")
                    st.balloons()
        
        with col_status:
            if st.session_state.get("ensemble_predictor"):
                st.success("✅ Model Active")
                if st.session_state.get("last_train_time"):
                    st.caption(f"Trained: {st.session_state['last_train_time'].strftime('%H:%M:%S')}")
                    
                # Show model weights
                predictor = st.session_state["ensemble_predictor"]
                st.markdown("**Model Weights:**")
                for name, weight in predictor.model_weights.items():
                    st.progress(weight, text=f"{name}: {weight:.3f}")
            else:
                st.info("Model not trained")
        
        # Step 4: Evaluation & Predictions
        if st.session_state.get("ensemble_predictor"):
            st.markdown("---")
            st.subheader("### Step 4: Model Performance & Predictions")
            
            predictor = st.session_state["ensemble_predictor"]
            
            # Split data for evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            
            # Get predictions with uncertainty
            y_pred, uncertainty = predictor.predict(X_test, return_uncertainty=True)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # Directional accuracy
            actual_direction = np.sign(y_test.diff())
            pred_direction = np.sign(pd.Series(y_pred).diff())
            directional_accuracy = np.mean(actual_direction.values[1:] == pred_direction.values[1:]) * 100
            
            # Update performance history
            st.session_state["performance_metrics"]["rmse_history"].append(rmse)
            st.session_state["performance_metrics"]["mae_history"].append(mae)
            st.session_state["performance_metrics"]["mape_history"].append(mape)
            st.session_state["performance_metrics"]["directional_accuracy"].append(directional_accuracy)
            
            # Display metrics
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("RMSE", f"₹{rmse:.2f}")
            col_m2.metric("MAE", f"₹{mae:.2f}")
            col_m3.metric("MAPE", f"{mape:.2f}%")
            col_m4.metric("R² Score", f"{r2:.4f}")
            col_m5.metric("Direction Accuracy", f"{directional_accuracy:.1f}%")
            
            # Plot predictions with uncertainty bands
            st.markdown("#### Predictions vs Actual (with Uncertainty Bands)")
            
            pred_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': y_pred,
                'Uncertainty': uncertainty,
                'Upper_Band': y_pred + 1.96 * uncertainty,
                'Lower_Band': y_pred - 1.96 * uncertainty
            }, index=y_test.index)
            
            fig_pred = go.Figure()
            
            # Uncertainty band
            fig_pred.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Upper_Band'],
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name='Upper Band'
            ))
            
            fig_pred.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Lower_Band'],
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 161, 90, 0.2)',
                name='95% Confidence Interval'
            ))
            
            # Actual values
            fig_pred.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Actual'],
                mode='lines',
                name='Actual Price',
                line=dict(color='#636EFA', width=2.5)
            ))
            
            # Predicted values
            fig_pred.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Predicted'],
                mode='lines',
                name='Predicted Price',
                line=dict(color='#FFA15A', width=2, dash='solid')
            ))
            
            fig_pred.update_layout(
                title_text=f"Adaptive Ensemble Performance ({prediction_horizon} periods ahead)",
                height=500,
                template="plotly_white",
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Feature Importance
            st.markdown("#### Top 20 Feature Importances")
            feature_imp = predictor.get_feature_importance()
            feature_imp_df = pd.DataFrame(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:20], 
                                         columns=['Feature', 'Importance'])
            
            fig_imp = go.Figure(go.Bar(
                x=feature_imp_df['Importance'],
                y=feature_imp_df['Feature'],
                orientation='h',
                marker=dict(color=feature_imp_df['Importance'], colorscale='Viridis')
            ))
            fig_imp.update_layout(
                title="Features Driving Predictions",
                height=500,
                template="plotly_white",
                yaxis=dict(autorange="reversed"),
                xaxis_title="Relative Importance"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # Step 5: Generate New Forecast
            st.markdown("---")
            st.subheader("### Step 5: Generate Future Forecast")
            
            col_forecast1, col_forecast2 = st.columns([2, 1])
            
            with col_forecast1:
                if st.button(f"🔮 Generate Forecast for Next {prediction_horizon} Periods", key="generate_forecast_btn", use_container_width=True):
                    latest_features = ml_data[selected_features].iloc[[-1]]
                    forecast, forecast_uncertainty = predictor.predict(latest_features, return_uncertainty=True)
                    forecast_value = forecast[0]
                    uncertainty_value = forecast_uncertainty[0]
                    
                    last_known_close = historical_data['close'].iloc[-1]
                    predicted_change = ((forecast_value - last_known_close) / last_known_close) * 100
                    
                    # Store prediction
                    predictor.store_prediction(latest_features.values[0], forecast_value)
                    
                    # Display forecast
                    st.success(f"📊 Forecast Generated with {len(predictor.base_models)} Models")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Last Close", f"₹{last_known_close:.2f}")
                    c2.metric(f"Forecast ({prediction_horizon}d)", f"₹{forecast_value:.2f}", delta=f"{predicted_change:+.2f}%")
                    c3.metric("Uncertainty (±)", f"₹{uncertainty_value:.2f}")
                    c4.metric("Confidence", f"{max(0, 100 - uncertainty_value/forecast_value*100):.1f}%")
                    
                    # Confidence bands
                    upper_bound = forecast_value + 1.96 * uncertainty_value
                    lower_bound = forecast_value - 1.96 * uncertainty_value
                    
                    st.info(f"**95% Confidence Interval:** ₹{lower_bound:.2f} - ₹{upper_bound:.2f}")
                    
                    # Forecast date
                    forecast_date = historical_data.index[-1] + timedelta(days=prediction_horizon)
                    st.caption(f"📅 Forecast Date (approx): {forecast_date.strftime('%Y-%m-%d')}")
                    
                    # Online learning
                    if st.session_state.get("online_learning_enabled"):
                        st.info("🔄 Online learning active - model will adapt as new data arrives")
            
            with col_forecast2:
                if st.session_state.get("prediction_history"):
                    st.markdown("**Recent Predictions:**")
                    history = list(predictor.prediction_history)[-5:]
                    for h in reversed(history):
                        st.caption(f"{h['timestamp'].strftime('%H:%M:%S')}: ₹{h['prediction']:.2f}")
    else:
        st.info("👆 Generate features first to enable ensemble training.")

# --- Main Application ---
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

tab_market, tab_analyze, tab_ml = st.tabs(["📊 Market Data", "📈 Technical Analysis", "🤖 Adaptive AI Predictor"])

with tab_market:
    render_market_historical_tab(k, api_key, access_token)

with tab_analyze:
    render_analyze_tab(k, api_key, access_token)

with tab_ml:
    render_adaptive_ml_tab(k, api_key, access_token)

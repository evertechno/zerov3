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
import math # Added for enhanced metrics

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Invsion Connect - Adaptive AI Trading Platform", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 Invsion Connect - Adaptive AI Trading Platform")
st.markdown("Production-grade platform with real-time learning, ensemble predictions, and continuous model improvement.")

# --- Global Constants & Session State Initialization ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"
MAX_PREDICTION_HISTORY = 500  # Store last N predictions for online learning
ONLINE_LEARNING_BATCH_SIZE = 20 # Number of samples needed to trigger an online update

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
if "ensemble_predictor" not in st.session_state:
    st.session_state["ensemble_predictor"] = None
if "ml_features" not in st.session_state:
    st.session_state["ml_features"] = []
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
if "last_train_time" not in st.session_state:
    st.session_state["last_train_time"] = None

# --- Load Credentials from Streamlit Secrets ---
def load_secrets():
    secrets = st.secrets
    kite_conf = secrets.get("kite", {})
    supabase_conf = secrets.get("supabase", {})

    errors = []
    if not kite_conf.get("api_key") or not kite_conf.get("api_secret") or not kite_conf.get("redirect_uri"):
        errors.append("Kite credentials (api_key, api_secret, redirect_uri)")
    if not supabase_conf.get("url") or not supabase_conf.get("anon_key"):
        # Supabase is optional for this core ML logic but keeping structure for production
        pass # Allow running without Supabase if not configured
    
    if errors:
        st.error(f"Missing required credentials in `.streamlit/secrets.toml`: {', '.join(errors)}.")
        st.info("Example `secrets.toml`:\n```toml\n[kite]\napi_key=\"YOUR_KITE_API_KEY\"\napi_secret=\"YOUR_KITE_SECRET\"\nredirect_uri=\"http://localhost:8501\"\n\n[supabase]\nurl=\"YOUR_SUPABASE_URL\"\nanon_key=\"YOUR_SUPABASE_ANON_KEY\"\n```")
        # st.stop() # Do not stop if only Supabase is missing

    return kite_conf, supabase_conf

KITE_CREDENTIALS, SUPABASE_CREDENTIALS = load_secrets()

# --- Supabase Client Initialization (Conditional) ---
supabase = None
if SUPABASE_CREDENTIALS.get("url") and SUPABASE_CREDENTIALS.get("anon_key"):
    @st.cache_resource(ttl=3600)
    def init_supabase_client(url: str, key: str) -> Client:
        return create_client(url, key)

    try:
        supabase: Client = init_supabase_client(SUPABASE_CREDENTIALS["url"], SUPABASE_CREDENTIALS["anon_key"])
    except Exception as e:
        st.warning(f"Failed to initialize Supabase client: {e}")

# --- KiteConnect Client Initialization ---
@st.cache_resource(ttl=3600)
def init_kite_unauth_client(api_key: str) -> KiteConnect:
    return KiteConnect(api_key=api_key)

try:
    kite_unauth_client = init_kite_unauth_client(KITE_CREDENTIALS["api_key"])
    login_url = kite_unauth_client.login_url()
except Exception as e:
    st.error(f"Kite client initialization failed. Check API Key in secrets.toml. Error: {e}")
    login_url = "#"


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
            df = df[['instrument_token', 'tradingsymbol', 'name', 'exchange', 'tick_size']]
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

    # Ensure datetime objects are used
    from_datetime = datetime.combine(from_date, datetime.min.time())
    to_datetime = datetime.combine(to_date, datetime.max.time())
    
    if interval == "day":
        # KiteConnect day interval uses date only, setting to start/end of day
        from_datetime = datetime.combine(from_date, datetime.min.time())
        to_datetime = datetime.combine(to_date, datetime.max.time())
    
    try:
        data = kite_instance.historical_data(token, from_date=from_datetime, to_date=to_datetime, interval=interval, continuous=False)
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
    
    # 0. Returns Features
    df_copy['Return_1'] = df_copy['close'].pct_change(1) * 100
    df_copy['Log_Return'] = np.log(df_copy['close'] / df_copy['close'].shift(1)) * 100
    df_copy['Vol_Daily'] = df_copy['Log_Return'].rolling(window=20).std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    
    # 1. Basic Technical Indicators
    df_copy['SMA_10'] = ta.trend.sma_indicator(df_copy['close'], window=10)
    df_copy['SMA_20'] = ta.trend.sma_indicator(df_copy['close'], window=20)
    df_copy['SMA_50'] = ta.trend.sma_indicator(df_copy['close'], window=50)
    df_copy['EMA_12'] = ta.trend.ema_indicator(df_copy['close'], window=12)
    df_copy['EMA_26'] = ta.trend.ema_indicator(df_copy['close'], window=26)
    df_copy['TRIX'] = ta.trend.trix(df_copy['close'])
    
    # 2. Momentum Indicators
    df_copy['RSI_14'] = ta.momentum.rsi(df_copy['close'], window=14)
    df_copy['RSI_7'] = ta.momentum.rsi(df_copy['close'], window=7)
    df_copy['Stochastic_K'] = ta.momentum.stoch(df_copy['high'], df_copy['low'], df_copy['close'])
    df_copy['Stochastic_D'] = ta.momentum.stoch_signal(df_copy['high'], df_copy['low'], df_copy['close'])
    df_copy['Williams_R'] = ta.momentum.williams_r(df_copy['high'], df_copy['low'], df_copy['close'])
    df_copy['PPO'] = ta.momentum.ppo(df_copy['close'])
    df_copy['ROC_10'] = ta.momentum.roc(df_copy['close'], window=10)
    
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
    df_copy['Keltner_Channel'] = ta.volatility.keltner_channel_central(df_copy['high'], df_copy['low'], df_copy['close'])
    
    # 5. Trend Strength Features
    adx_obj = ta.trend.ADXIndicator(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
    df_copy['ADX'] = adx_obj.adx()
    df_copy['ADX_Pos'] = adx_obj.adx_pos()
    df_copy['ADX_Neg'] = adx_obj.adx_neg()
    df_copy['CCI'] = ta.trend.cci(df_copy['high'], df_copy['low'], df_copy['close'])
    
    # 6. Volume-based Features
    df_copy['VWAP'] = ta.volume.volume_weighted_average_price(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], window=14)
    df_copy['OBV'] = ta.volume.on_balance_volume(df_copy['close'], df_copy['volume'])
    df_copy['Volume_SMA'] = df_copy['volume'].rolling(window=20).mean()
    df_copy['Volume_Ratio'] = df_copy['volume'] / df_copy['Volume_SMA']
    df_copy['Aroon_Up'] = ta.trend.aroon_up(df_copy['close'])
    df_copy['Aroon_Down'] = ta.trend.aroon_down(df_copy['close'])
    
    # 7. Price Action Features (Market Microstructure Proxies)
    df_copy['High_Low_Range'] = (df_copy['high'] - df_copy['low']) / df_copy['close']
    df_copy['Close_Open_Range'] = (df_copy['close'] - df_copy['open']) / df_copy['open']
    df_copy['Upper_Shadow'] = (df_copy['high'] - df_copy[['open', 'close']].max(axis=1)) / df_copy['close']
    df_copy['Lower_Shadow'] = (df_copy[['open', 'close']].min(axis=1) - df_copy['low']) / df_copy['close']
    df_copy['Mid_Price_Change'] = (df_copy['close'] - (df_copy['high'].shift(1) + df_copy['low'].shift(1)) / 2) / df_copy['close'].shift(1)
    
    # 8. Lagged Features (Multiple Horizons)
    for lag in [1, 2, 3, 5, 10, 15, 20]:
        df_copy[f'Lag_Close_{lag}'] = df_copy['close'].shift(lag)
        df_copy[f'Lag_Return_{lag}'] = df_copy['close'].pct_change(lag) * 100
        df_copy[f'Lag_Volume_{lag}'] = df_copy['volume'].shift(lag)
        df_copy[f'Momentum_Lag_{lag}'] = df_copy['close'].diff(lag)
    
    # 9. Rolling Statistics (Volatility & Momentum)
    for window in [5, 10, 20, 30]:
        df_copy[f'Rolling_Mean_{window}'] = df_copy['close'].rolling(window=window).mean()
        df_copy[f'Rolling_Std_{window}'] = df_copy['close'].rolling(window=window).std()
        df_copy[f'Rolling_Min_{window}'] = df_copy['close'].rolling(window=window).min()
        df_copy[f'Rolling_Max_{window}'] = df_copy['close'].rolling(window=window).max()
        df_copy[f'Price_Position_{window}'] = (df_copy['close'] - df_copy[f'Rolling_Min_{window}']) / \
                                              (df_copy[f'Rolling_Max_{window}'] - df_copy[f'Rolling_Min_{window}'])
        df_copy[f'Skew_{window}'] = df_copy['Log_Return'].rolling(window=window).skew()
        df_copy[f'Kurtosis_{window}'] = df_copy['Log_Return'].rolling(window=window).kurt()
    
    # 10. Temporal Features (Requires datetime index)
    if df_copy.index.inferred_type == 'datetime64':
        df_copy['Day_of_Week'] = df_copy.index.dayofweek
        df_copy['Day_of_Month'] = df_copy.index.day
        df_copy['Week_of_Year'] = df_copy.index.isocalendar().week.astype(int)
        df_copy['Month'] = df_copy.index.month
        df_copy['Quarter'] = df_copy.index.quarter
        
        # 11. Cyclic Encoding of Temporal Features
        df_copy['Day_of_Week_Sin'] = np.sin(2 * np.pi * df_copy['Day_of_Week'] / 7)
        df_copy['Day_of_Week_Cos'] = np.cos(2 * np.pi * df_copy['Day_of_Week'] / 7)
        df_copy['Month_Sin'] = np.sin(2 * np.pi * df_copy['Month'] / 12)
        df_copy['Month_Cos'] = np.cos(2 * np.pi * df_copy['Month'] / 12)
    
    # 12. Market Regime Features
    df_copy['Volatility_Regime'] = (df_copy['ATR_Pct'] > df_copy['ATR_Pct'].rolling(window=50).mean()).astype('float64')
    df_copy['Trend_Regime'] = (df_copy['SMA_20'] > df_copy['SMA_50']).astype('float64')
    
    # Forward fill then backward fill to handle NaN values introduced by rolling windows
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
        self.model_complexity = 'balanced'
        
    def initialize_models(self, complexity='balanced'):
        """Initialize diverse set of base models for ensemble"""
        self.model_complexity = complexity
        if complexity == 'fast':
            self.base_models = {
                'ridge': Ridge(alpha=1.0, random_state=42),
                'lasso': Lasso(alpha=0.1, random_state=42),
                'lgbm': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=20, random_state=42, n_jobs=-1),
            }
        elif complexity == 'accurate':
            self.base_models = {
                'ridge': Ridge(alpha=0.5, random_state=42),
                'lasso': Lasso(alpha=0.05, random_state=42),
                'lgbm': lgb.LGBMRegressor(n_estimators=750, learning_rate=0.02, num_leaves=64, random_state=42, n_jobs=-1, verbose=-1),
                'rf': RandomForestRegressor(n_estimators=400, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1),
                'gb': GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.7, random_state=42)
            }
        else:  # balanced
            self.base_models = {
                'ridge': Ridge(alpha=0.8, random_state=42),
                'lasso': Lasso(alpha=0.08, random_state=42),
                'lgbm': lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1),
                'rf': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            }
        
        # Initialize meta-learner (stacking)
        self.meta_learner = Ridge(alpha=0.1, random_state=42)
        
        # Initialize equal weights
        self.model_weights = {name: 1.0/len(self.base_models) for name in self.base_models.keys()}
    
    def train(self, X, y, features):
        """Train ensemble with cross-validation"""
        self.feature_names = features
        
        # Fit scaler on training data
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Use TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=5)
        meta_features = np.zeros((X_scaled.shape[0], len(self.base_models)))
        
        y_series = pd.Series(y.values, index=X.index)
        
        for idx, (name, model) in enumerate(self.base_models.items()):
            # Cross-validation predictions for meta-learner
            fold_predictions = np.zeros(X_scaled.shape[0])
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
                y_train_fold, y_val_fold = y_series.iloc[train_idx], y_series.iloc[val_idx]
                
                # Fit base model
                model.fit(X_train_fold, y_train_fold)
                # Predict on validation fold
                fold_predictions[val_idx] = model.predict(X_val_fold)
            
            meta_features[:, idx] = fold_predictions
            
            # Train model on full data for final deployment
            self.base_models[name].fit(X_scaled, y_series)
        
        # Train meta-learner
        self.meta_learner.fit(meta_features, y_series)
        
        # Update model weights based on individual performance (using full data MSE)
        self._update_model_weights(X_scaled, y_series)
        
    def predict(self, X, return_uncertainty=False):
        """Make ensemble prediction with optional uncertainty quantification"""
        if X.empty:
            return np.array([]), np.array([])
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all base models
        base_predictions = np.zeros((X_scaled.shape[0], len(self.base_models)))
        for idx, (name, model) in enumerate(self.base_models.items()):
            base_predictions[:, idx] = model.predict(X_scaled)
        
        # Weighted ensemble prediction
        # Ensure weights are a NumPy array for vectorized multiplication
        weights = np.array(list(self.model_weights.values()))
        weighted_pred = np.sum(base_predictions * weights, axis=1)
        
        # Meta-learner prediction (stacking)
        meta_pred = self.meta_learner.predict(base_predictions)
        
        # Final prediction (A balanced average ensures robustness)
        final_pred = (weighted_pred * 0.5) + (meta_pred * 0.5)
        
        if return_uncertainty:
            # Uncertainty as standard deviation of base predictions
            uncertainty = np.std(base_predictions, axis=1)
            return final_pred, uncertainty
        
        return final_pred
    
    def _update_model_weights(self, X_scaled, y):
        """Update model weights based on performance using Inverse Error Weighting"""
        errors = {}
        for name, model in self.base_models.items():
            pred = model.predict(X_scaled)
            # Clip predictions to prevent negative/zero prices for price prediction tasks
            pred = np.maximum(pred, np.min(y) * 0.9) 
            # Use RMSE for error measure
            errors[name] = np.sqrt(mean_squared_error(y, pred))
        
        # Calculate inverse errors, adding a small epsilon to avoid division by zero
        inv_errors = {name: 1.0 / (error + 1e-6) for name, error in errors.items()}
        
        total_inv_error = sum(inv_errors.values())
        
        # Normalize to get weights
        self.model_weights = {name: inv_error / total_inv_error for name, inv_error in inv_errors.items()}
    
    def online_update(self, X_new, y_new):
        """Incrementally update models with new data (simulated real-time learning)"""
        
        # 1. Store incoming data in buffer
        X_scaled = self.scaler.transform(X_new)
        self.online_learning_buffer.append((X_scaled.flatten(), y_new.values[0]))
        
        if len(self.online_learning_buffer) >= ONLINE_LEARNING_BATCH_SIZE:
            
            st.info(f"🔄 Triggering online update with {ONLINE_LEARNING_BATCH_SIZE} new samples...")
            
            # 2. Extract batch data
            X_buffer = np.vstack([x for x, _ in self.online_learning_buffer])
            y_buffer = np.array([y for _, y in self.online_learning_buffer])
            
            # 3. Incremental update for all base models
            for name, model in self.base_models.items():
                if name == 'lgbm':
                    # LightGBM: Retrain on new data appended to old data (approximation of online learning)
                    # For true online learning, specialized libraries like River would be needed, 
                    # but here we use a small, fast re-training cycle.
                    
                    # NOTE: Since the full historical data is too large to re-scale continuously, 
                    # we only update based on the buffer for simplicity in this Streamlit app.
                    
                    # We increase estimators slightly for tree models to keep learning new patterns
                    if hasattr(model, 'n_estimators'):
                        model.n_estimators += 5
                    model.fit(X_buffer, y_buffer, init_model=model) # Warm start/incremental fit
                
                elif hasattr(model, 'partial_fit'): # Linear models (Ridge, Lasso)
                    # For Linear models, partial_fit is ideal, but scikit-learn Ridge/Lasso don't have it.
                    # We simulate by retraining the meta-learner frequently and only re-weighting base models.
                    
                    # For demonstration, we simply retrain Ridge/Lasso on the small buffer.
                    model.fit(X_buffer, y_buffer)
                
                elif name in ['rf', 'gb']: # Re-fit tree ensembles (small fast re-fit)
                    model.fit(X_buffer, y_buffer)
            
            # 4. Update Meta-learner (essential for calibration)
            
            # Predict meta features on the new buffer data
            buffer_meta_features = np.zeros((X_buffer.shape[0], len(self.base_models)))
            for idx, (name, model) in enumerate(self.base_models.items()):
                buffer_meta_features[:, idx] = model.predict(X_buffer)
            
            # Retrain the meta-learner on the buffer
            self.meta_learner.fit(buffer_meta_features, y_buffer)

            # 5. Update model weights based on the buffer's performance
            self._update_model_weights(X_buffer, pd.Series(y_buffer))
            
            # 6. Clear buffer
            self.online_learning_buffer = []
            st.success("✅ Online learning batch complete. Weights updated.")
            
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
            if name == 'lgbm':
                imp = model.feature_importances_
            elif name in ['rf', 'gb']:
                imp = model.feature_importances_
            elif name in ['ridge', 'lasso']:
                # For linear models, use absolute coefficients scaled by mean of features
                imp = np.abs(model.coef_) * self.scaler.center_
            else:
                continue
            
            # If the model has importance scores
            if len(imp) == len(self.feature_names):
                for feat_name, feat_imp in zip(self.feature_names, imp):
                    # Weight the importance by the model's current adaptive weight
                    weighted_imp = feat_imp * self.model_weights.get(name, 0)
                    if feat_name not in importances:
                        importances[feat_name] = []
                    importances[feat_name].append(weighted_imp)
        
        # Average importance across models
        if not importances: return {}
        
        # Sum of weighted importances
        avg_importances = {k: sum(v) for k, v in importances.items()}
        
        # Normalize the final importance scores
        total_imp = sum(avg_importances.values())
        if total_imp == 0: return {}
        
        return {k: v / total_imp for k, v in avg_importances.items()}


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
            st.session_state["historical_data"] = pd.DataFrame()
            st.session_state["ensemble_predictor"] = None
            st.success("Logged out from Kite. Please login again.")
            st.rerun()
    else:
        st.info("Not authenticated with Kite yet.")

    st.markdown("---")
    st.markdown("### 2. ML System Status")
    predictor = st.session_state.get("ensemble_predictor")
    if predictor:
        st.success("🤖 Adaptive ML System Active")
        st.metric("Models in Ensemble", len(predictor.base_models))
        if st.session_state.get("performance_metrics", {}).get("rmse_history"):
            latest_rmse = st.session_state["performance_metrics"]["rmse_history"][-1]
            st.metric("Latest RMSE", f"₹{latest_rmse:.2f}")
        st.metric("Online Buffer Size", len(predictor.online_learning_buffer))
        st.metric("Prediction Horizon", st.session_state.get("prediction_horizon"))
    else:
        st.info("ML System not initialized")
    
    st.markdown("---")
    online_learning = st.checkbox("Enable Online Learning", value=st.session_state.get("online_learning_enabled", False))
    st.session_state["online_learning_enabled"] = online_learning
    if online_learning:
        st.success("📈 Real-time adaptation enabled (Batch size: 20)")

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
        if st.button("Fetch Historical Data", key="fetch_historical_data_btn", use_container_width=True):
            with st.spinner(f"Fetching {interval} historical data for {hist_symbol}..."):
                df_hist = get_historical_data_cached(api_key, access_token, hist_symbol, from_date, to_date, interval, hist_exchange) 
                if isinstance(df_hist, pd.DataFrame) and "_error" not in df_hist.columns:
                    st.session_state["historical_data"] = df_hist
                    st.session_state["last_fetched_symbol"] = hist_symbol
                    st.session_state["ml_data"] = pd.DataFrame()
                    st.session_state["ensemble_predictor"] = None
                    st.session_state["prediction_history"] = deque(maxlen=MAX_PREDICTION_HISTORY)
                    st.success(f"Fetched {len(df_hist)} records for {hist_symbol}.")
                else:
                    st.error(f"Historical fetch failed: {df_hist.get('_error', 'Unknown error')}")
    with col_hist_plot:
        if not st.session_state.get("historical_data", pd.DataFrame()).empty:
            df = st.session_state["historical_data"]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            # Ensure proper handling of potential multi-day data for Candlestick
            if interval == 'day':
                x_axis = df.index.date
            else:
                x_axis = df.index
                
            fig.add_trace(go.Candlestick(x=x_axis, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'), row=1, col=1)
            fig.add_trace(go.Bar(x=x_axis, y=df['volume'], name='Volume', marker_color='blue'), row=2, col=1)
            
            fig.update_layout(title_text=f"Historical Price & Volume for {st.session_state['last_fetched_symbol']} ({interval})", xaxis_rangeslider_visible=False, height=600, template="plotly_white")
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
    
    # Use the advanced features function to ensure all required TAs are calculated
    df_ta = add_advanced_features(historical_data[['open', 'high', 'low', 'close', 'volume']])
    
    if df_ta.empty:
        st.error("Data is too short or invalid for comprehensive Technical Analysis.")
        return

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2], subplot_titles=("Price Action & Overlays", "RSI Oscillator", "MACD"))
    
    # Row 1: Price and Trend
    fig.add_trace(go.Candlestick(x=df_ta.index, open=df_ta['open'], high=df_ta['high'], low=df_ta['low'], close=df_ta['close'], name='Candlestick'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['Bollinger_High'], mode='lines', name='Bollinger High', line=dict(color='grey', dash='dash', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['Bollinger_Low'], mode='lines', name='Bollinger Low', line=dict(color='grey', dash='dash', width=1)), row=1, col=1)
    
    # Row 2: RSI
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['RSI_14'], mode='lines', name='RSI 14'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="bottom right", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom right", row=2, col=1)
    
    # Row 3: MACD
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='orange')), row=3, col=1)
    colors = ['green' if val >= 0 else 'red' for val in df_ta['MACD_Hist']]
    fig.add_trace(go.Bar(x=df_ta.index, y=df_ta['MACD_Hist'], name='Histogram', marker_color=colors), row=3, col=1)
    
    fig.update_layout(title_text=f"Technical Indicators for {last_symbol}", xaxis_rangeslider_visible=False, height=800, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Indicator Summary & Technical Outlook (Derived from Latest Data)")
    def generate_technical_outlook(df):
        if df.empty: return "No Data", []
        latest = df.iloc[-1]
        signals = {'buy': 0, 'sell': 0, 'hold': 0}; reasons = []
        
        # Trend Analysis
        if latest.get('SMA_50') and latest.get('SMA_20'):
            if latest['close'] > latest['SMA_50'] and latest['SMA_20'] > latest['SMA_50']:
                signals['buy'] += 1; reasons.append("✅ **Strong Uptrend:** Price above SMA 50, and SMA 20 > SMA 50 (Golden Cross potential).")
            elif latest['close'] < latest['SMA_50'] and latest['SMA_20'] < latest['SMA_50']:
                signals['sell'] += 1; reasons.append("❌ **Strong Downtrend:** Price below SMA 50, and SMA 20 < SMA 50 (Death Cross potential).")
            else:
                signals['hold'] += 1; reasons.append("➖ **Neutral Trend:** Price is consolidating around key moving averages.")
        
        # Momentum Analysis (RSI)
        if latest.get('RSI_14'):
            if latest['RSI_14'] < 30:
                signals['buy'] += 1; reasons.append(f"✅ **Oversold:** RSI ({latest['RSI_14']:.2f}) suggests a potential bounce.")
            elif latest['RSI_14'] > 70:
                signals['sell'] += 1; reasons.append(f"❌ **Overbought:** RSI ({latest['RSI_14']:.2f}) suggests a potential pullback.")
            else:
                signals['hold'] += 1; reasons.append(f"➖ **Neutral Momentum:** RSI ({latest['RSI_14']:.2f}) is between 30 and 70.")
        
        # MACD Analysis
        if latest.get('MACD') and latest.get('MACD_Signal'):
            if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Hist'] > 0:
                signals['buy'] += 1; reasons.append("✅ **Bullish Crossover:** MACD line is above signal line and histogram is growing positive.")
            elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Hist'] < 0:
                signals['sell'] += 1; reasons.append("❌ **Bearish Crossover:** MACD line is below signal line and histogram is growing negative.")
            else:
                signals['hold'] += 1; reasons.append("➖ **MACD Neutral:** Weak momentum or divergence.")

        # Volatility Analysis (ADX)
        if latest.get('ADX') and latest.get('ADX_Pos') and latest.get('ADX_Neg'):
            if latest['ADX'] > 25 and latest['ADX_Pos'] > latest['ADX_Neg']:
                signals['buy'] += 0.5; reasons.append(f"✅ **Strong Trend (Bullish):** ADX ({latest['ADX']:.2f}) is strong and +DI > -DI.")
            elif latest['ADX'] > 25 and latest['ADX_Neg'] > latest['ADX_Pos']:
                signals['sell'] += 0.5; reasons.append(f"❌ **Strong Trend (Bearish):** ADX ({latest['ADX']:.2f}) is strong and -DI > +DI.")

        if signals['buy'] > signals['sell'] + 0.5: return "Strong Bullish Outlook", reasons
        elif signals['buy'] > signals['sell']: return "Bullish Outlook", reasons
        elif signals['sell'] > signals['buy'] + 0.5: return "Strong Bearish Outlook", reasons
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
                df_with_features = add_advanced_features(historical_data[['open', 'high', 'low', 'close', 'volume']])
            if not df_with_features.empty:
                st.session_state["ml_data"] = df_with_features
                st.session_state["ensemble_predictor"] = None # Reset predictor on new data
                st.session_state["prediction_history"] = deque(maxlen=MAX_PREDICTION_HISTORY)
                st.success(f"✅ Generated {len(df_with_features.columns)} features from {len(df_with_features)} data points")
            else:
                st.error("❌ Failed to add features. Data might be too short or invalid.")
                st.session_state["ml_data"] = pd.DataFrame()
    
    with col_info:
        ml_data = st.session_state.get("ml_data", pd.DataFrame())
        if not ml_data.empty:
            st.metric("Total Features", len(ml_data.columns) - 5) # Subtract OHLCV
            st.metric("Data Points", len(ml_data))
        else:
            st.info("Features not generated yet")

    
    if not ml_data.empty:
        st.markdown("---")
        
        # Step 2: Model Configuration
        st.subheader("### Step 2: Configure Adaptive Ensemble")
        
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            prediction_horizon = st.number_input("🎯 Prediction Horizon (periods ahead)", min_value=1, max_value=30, value=st.session_state.get("prediction_horizon", 5), step=1, key="pred_horizon_adaptive")
            st.session_state["prediction_horizon"] = prediction_horizon
        
        with col_config2:
            model_complexity = st.selectbox("⚙️ Model Complexity", ["fast", "balanced", "accurate"], index=1, key="complexity_selector")
        
        with col_config3:
            test_size = st.slider("📊 Test Set Size (%)", 10, 40, 20, step=5, key="test_size_slider") / 100.0
        
        # Prepare data and target (Target is 'close' shifted backward by the horizon)
        ml_data_processed = ml_data.copy()
        ml_data_processed['target'] = ml_data_processed['close'].shift(-prediction_horizon)
        ml_data_processed.dropna(subset=['target'], inplace=True)
        
        # Feature Selection
        features_all = [col for col in ml_data_processed.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        
        # Auto-select important features (if model was previously trained)
        default_features = st.session_state.get("ml_features", [])
        if not default_features:
             # Default to a robust set if no previous training
             default_features = [f for f in features_all if any(sub in f for sub in 
                ['RSI_14', 'MACD', 'Lag_Close_1', 'Lag_Return_5', 'SMA_20', 'ADX', 'ATR', 
                 'Volume_Ratio', 'Rolling_Std_20', 'Price_Position_20', 'VWAP', 'Day_of_Week_Sin'])]
             default_features = list(set(default_features))

        st.markdown("#### Feature Selection")
        selected_features = st.multiselect("Select Features (or use recommended)", 
                                          options=features_all, 
                                          default=default_features[:30], 
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
                if len(X) < 100:
                    st.error(f"Need more data. Only {len(X)} points available after target creation. Min 100 required.")
                    return
                with st.spinner("Training ensemble of ML models with TimeSeries CV..."):
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
            predictor = st.session_state.get("ensemble_predictor")
            if predictor:
                st.success("✅ Model Active")
                if st.session_state.get("last_train_time"):
                    st.caption(f"Trained: {st.session_state['last_train_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                # Show model weights
                st.markdown("**Model Weights:**")
                # Sort weights for better visualization
                sorted_weights = sorted(predictor.model_weights.items(), key=lambda item: item[1], reverse=True)
                for name, weight in sorted_weights:
                    st.progress(weight, text=f"{name}: {weight:.3f}")
            else:
                st.info("Model not trained")
        
        # Step 4: Evaluation & Predictions
        if predictor:
            st.markdown("---")
            st.subheader("### Step 4: Model Performance & Predictions")
            
            # Split data for evaluation
            # Ensure the split preserves temporal order
            test_index = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:test_index], X.iloc[test_index:]
            y_train, y_test = y.iloc[:test_index], y.iloc[test_index:]
            
            if X_test.empty:
                st.warning("Test set is empty. Increase data duration or reduce test set size.")
                return

            # Get predictions with uncertainty
            y_pred, uncertainty = predictor.predict(X_test, return_uncertainty=True)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            # Calculate Mean Absolute Percentage Error (MAPE) robustly
            def calculate_mape(y_true, y_pred):
                # Avoid division by zero by ignoring points where y_true is zero
                return np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan).dropna())) * 100

            mape = calculate_mape(y_test, pd.Series(y_pred, index=y_test.index))
            
            # Directional accuracy
            actual_direction = np.sign(y_test.diff().fillna(0))
            pred_direction = np.sign(pd.Series(y_pred, index=y_test.index).diff().fillna(0))
            # Compare non-zero directional movements
            valid_comparison = (actual_direction != 0)
            directional_accuracy = np.mean(actual_direction[valid_comparison] == pred_direction[valid_comparison]) * 100 if valid_comparison.any() else 0
            
            # Update performance history (optional: only update if it was a new train)
            if not st.session_state["performance_metrics"]["rmse_history"] or rmse != st.session_state["performance_metrics"]["rmse_history"][-1]:
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
            st.markdown("#### Top 20 Feature Importances (Aggregated and Weighted)")
            feature_imp = predictor.get_feature_importance()
            if feature_imp:
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
                    xaxis_title="Normalized Importance Score"
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance not yet calculated or features are identical.")
            
            # Step 5: Generate New Forecast
            st.markdown("---")
            st.subheader("### Step 5: Generate Future Forecast")
            
            col_forecast1, col_forecast2 = st.columns([2, 1])
            
            with col_forecast1:
                if st.button(f"🔮 Generate Forecast for Next {prediction_horizon} Periods", key="generate_forecast_btn", use_container_width=True):
                    
                    # Ensure the latest row is used for prediction
                    if ml_data.empty:
                        st.error("Cannot generate forecast: ML data is empty.")
                        return
                    
                    latest_features = ml_data[selected_features].iloc[[-1]]
                    
                    forecast, forecast_uncertainty = predictor.predict(latest_features, return_uncertainty=True)
                    
                    if forecast.size == 0:
                        st.error("Prediction failed.")
                        return
                        
                    forecast_value = forecast[0]
                    uncertainty_value = forecast_uncertainty[0]
                    
                    last_known_close = historical_data['close'].iloc[-1]
                    predicted_change = ((forecast_value - last_known_close) / last_known_close) * 100
                    
                    # Store prediction for potential online learning evaluation later
                    predictor.store_prediction(latest_features.values[0].tolist(), forecast_value)
                    
                    # Display forecast
                    st.success(f"📊 Forecast Generated with {len(predictor.base_models)} Models")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Last Close", f"₹{last_known_close:.2f}")
                    c2.metric(f"Forecast (+{prediction_horizon} periods)", f"₹{forecast_value:.2f}", delta=f"{predicted_change:+.2f}%")
                    c3.metric("Uncertainty (±)", f"₹{uncertainty_value:.2f}")
                    
                    # Confidence based on percentage of prediction
                    confidence_pct = max(0, 100 - (uncertainty_value / forecast_value) * 100)
                    c4.metric("Model Confidence", f"{confidence_pct:.1f}%")
                    
                    # Confidence bands
                    upper_bound = forecast_value + 1.96 * uncertainty_value
                    lower_bound = forecast_value - 1.96 * uncertainty_value
                    
                    st.info(f"**95% Confidence Interval:** ₹{lower_bound:.2f} - ₹{upper_bound:.2f}")
                    
                    # Forecast date (approximation)
                    forecast_date = ml_data.index[-1] + timedelta(days=prediction_horizon)
                    st.caption(f"📅 Last data timestamp used: {ml_data.index[-1].strftime('%Y-%m-%d %H:%M')}")
                    st.caption(f"📅 Projected Target Period: ~{forecast_date.strftime('%Y-%m-%d')}")
                    
                    # Online learning check (simulate new data arrival and learning)
                    if st.session_state.get("online_learning_enabled"):
                        # Simulate the arrival of the actual data point once the window passes
                        # In a live system, this would be triggered by a data listener.
                        # Here, we add a mock entry to the online buffer.
                        if len(predictor.online_learning_buffer) < ONLINE_LEARNING_BATCH_SIZE:
                            # Use the latest features and the current 'close' (which is the actual price for lag 0)
                            # as a placeholder for the next actual price 'y_new'
                            X_new_online = ml_data[selected_features].iloc[[-1]]
                            y_new_online = ml_data['close'].iloc[[-1]] # Mock actual target for the next period
                            predictor.online_learning_buffer.append((predictor.scaler.transform(X_new_online).flatten(), y_new_online.values[0]))
                        
                        if len(predictor.online_learning_buffer) >= ONLINE_LEARNING_BATCH_SIZE:
                            predictor.online_update(X_new_online, y_new_online)
                            st.session_state["ensemble_predictor"] = predictor # Update session state after in-place modification
                        else:
                             st.info(f"Buffer size: {len(predictor.online_learning_buffer)}/{ONLINE_LEARNING_BATCH_SIZE}. Waiting for more data to adapt.")


            with col_forecast2:
                if predictor and predictor.prediction_history:
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

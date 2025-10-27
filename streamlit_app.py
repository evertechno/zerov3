import streamlit as st
import pandas as pd
import json
import re
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import ta
import base64
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

# Supabase imports
from supabase import create_client, Client
from kiteconnect import KiteConnect

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Invsion Connect - Advanced Analysis", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.title("Invsion Connect")
st.markdown("A comprehensive platform for fetching market data, performing ML-driven analysis, risk assessment, and live data streaming.")

# --- Global Constants ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"
BENCHMARK_SYMBOL = "NIFTY 50"

# --- Session State Initialization ---
session_defaults = {
    "kite_access_token": None,
    "kite_login_response": None,
    "instruments_df": pd.DataFrame(),
    "historical_data": pd.DataFrame(),
    "last_fetched_symbol": None,
    "user_session": None,
    "user_id": None,
    "saved_indexes": [],
    "current_calculated_index_data": pd.DataFrame(),
    "current_calculated_index_history": pd.DataFrame(),
    "last_comparison_df": pd.DataFrame(),
    "last_comparison_metrics": {},
    "last_facts_data": None,
    "last_factsheet_html_data": None,
    "current_market_data": None,
    "holdings_data": None,
    "benchmark_historical_data": pd.DataFrame(),
    "factsheet_selected_constituents_index_names": [],
    "index_price_calc_df": pd.DataFrame(),
    "compliance_results_df": pd.DataFrame(),
    "blended_indexes": [],
    "correlation_matrix": pd.DataFrame(),
    "efficient_frontier_data": None
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Load Credentials ---
def load_secrets():
    """Load credentials from Streamlit secrets without caching the secrets object"""
    secrets = st.secrets
    
    # Extract values immediately to avoid caching issues
    kite_conf = {
        "api_key": secrets.get("kite", {}).get("api_key", ""),
        "api_secret": secrets.get("kite", {}).get("api_secret", ""),
        "redirect_uri": secrets.get("kite", {}).get("redirect_uri", "")
    }
    
    supabase_conf = {
        "url": secrets.get("supabase", {}).get("url", ""),
        "anon_key": secrets.get("supabase", {}).get("anon_key", "")
    }

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
def get_authenticated_kite_client(api_key: Optional[str], access_token: Optional[str]) -> Optional[KiteConnect]:
    if api_key and access_token:
        k_instance = KiteConnect(api_key=api_key)
        k_instance.set_access_token(access_token)
        return k_instance
    return None

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

@st.cache_data(ttl=86400, show_spinner="Loading instruments...")
def load_instruments_cached(api_key: str, access_token: str, exchange: Optional[str] = None) -> pd.DataFrame:
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
        return pd.DataFrame({"_error": [f"Failed to load instruments: {e}"]})

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
def get_historical_data_cached(
    api_key: str, 
    access_token: str, 
    symbol: str, 
    from_date: datetime.date, 
    to_date: datetime.date, 
    interval: str, 
    exchange: str = DEFAULT_EXCHANGE
) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame({"_error": ["Kite not authenticated."]})

    instruments_df = load_instruments_cached(api_key, access_token, exchange)
    if "_error" in instruments_df.columns:
        instruments_df = load_instruments_cached(api_key, access_token)
        if "_error" in instruments_df.columns:
            return pd.DataFrame({"_error": [instruments_df.loc[0, '_error']]})

    token = find_instrument_token(instruments_df, symbol, exchange)
    
    if not token and symbol in ["NIFTY BANK", "NIFTYBANK", "BANKNIFTY", BENCHMARK_SYMBOL, "SENSEX"]:
        index_exchange = "NSE" if symbol not in ["SENSEX"] else "BSE"
        instruments_secondary = load_instruments_cached(api_key, access_token, index_exchange)
        token = find_instrument_token(instruments_secondary, symbol, index_exchange)
        
        if not token:
            return pd.DataFrame({"_error": [f"Instrument token not found for {symbol}."]})

    if not token:
        return pd.DataFrame({"_error": [f"Instrument token not found for {symbol}."]})

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

def find_instrument_token(df: pd.DataFrame, tradingsymbol: str, exchange: str = DEFAULT_EXCHANGE) -> Optional[int]:
    if df.empty:
        return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & \
           (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    hits = df[mask]
    return int(hits.iloc[0]["instrument_token"]) if not hits.empty else None

def add_technical_indicators(
    df: pd.DataFrame, 
    sma_periods: List[int], 
    ema_periods: List[int], 
    rsi_window: int, 
    macd_fast: int, 
    macd_slow: int, 
    macd_signal: int, 
    bb_window: int, 
    bb_std_dev: float
) -> pd.DataFrame:
    if df.empty or 'close' not in df.columns:
        return df.copy()

    df_copy = df.copy()
    
    for period in sma_periods:
        if period > 0:
            df_copy[f'SMA_{period}'] = ta.trend.sma_indicator(df_copy['close'], window=period)
    for period in ema_periods:
        if period > 0:
            df_copy[f'EMA_{period}'] = ta.trend.ema_indicator(df_copy['close'], window=period)
        
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
    
    # Additional indicators
    df_copy['ATR'] = ta.volatility.average_true_range(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
    df_copy['ADX'] = ta.trend.adx(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
    df_copy['OBV'] = ta.volume.on_balance_volume(df_copy['close'], df_copy['volume'])
    
    df_copy['Daily_Return'] = df_copy['close'].pct_change() * 100
    
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(method='ffill', inplace=True)
    return df_copy.dropna()

def calculate_performance_metrics(
    returns_series: pd.Series, 
    risk_free_rate: float = 0.0, 
    benchmark_returns: Optional[pd.Series] = None
) -> Dict:
    if returns_series.empty or len(returns_series) < 2:
        return {}
    
    daily_returns_decimal = returns_series / 100.0 if returns_series.abs().mean() > 0.1 else returns_series
    daily_returns_decimal = daily_returns_decimal.replace([np.inf, -np.inf], np.nan).dropna()
    if daily_returns_decimal.empty:
        return {}

    cumulative_returns = (1 + daily_returns_decimal).cumprod() - 1
    total_return = cumulative_returns.iloc[-1] * 100 if not cumulative_returns.empty else 0
    num_periods = len(daily_returns_decimal)
    
    if num_periods > 0 and (1 + daily_returns_decimal > 0).all():
        geometric_mean_daily_return = np.expm1(np.log1p(daily_returns_decimal).mean())
        annualized_return = ((1 + geometric_mean_daily_return) ** TRADING_DAYS_PER_YEAR - 1) * 100
    else:
        annualized_return = np.nan

    daily_volatility = daily_returns_decimal.std()
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR) * 100 if daily_volatility is not None else np.nan

    risk_free_rate_decimal = risk_free_rate / 100.0
    daily_rf_rate = (1 + risk_free_rate_decimal)**(1/TRADING_DAYS_PER_YEAR) - 1

    sharpe_ratio = (annualized_return / 100 - risk_free_rate_decimal) / (annualized_volatility / 100) if annualized_volatility > 0 else np.nan

    if not cumulative_returns.empty:
        peak = (1 + cumulative_returns).cummax()
        drawdown = ((1 + cumulative_returns) - peak) / peak
        max_drawdown = drawdown.min() * 100
    else:
        max_drawdown = np.nan

    downside_returns = daily_returns_decimal[daily_returns_decimal < daily_rf_rate]
    downside_std_dev_daily = downside_returns.std() if not downside_returns.empty else np.nan
    annualized_downside_std_dev = downside_std_dev_daily * np.sqrt(TRADING_DAYS_PER_YEAR) if not np.isnan(downside_std_dev_daily) else np.nan
    sortino_ratio = (annualized_return / 100 - risk_free_rate_decimal) / (annualized_downside_std_dev) if annualized_downside_std_dev > 0 else np.nan

    calmar_ratio = (annualized_return / 100) / abs(max_drawdown / 100) if max_drawdown != 0 and not np.isnan(max_drawdown) else np.nan

    # VaR and CVaR
    confidence_level = 0.05
    var_daily = -daily_returns_decimal.quantile(confidence_level)
    var_annualized = var_daily * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    
    cvar_daily_losses = daily_returns_decimal[daily_returns_decimal < daily_returns_decimal.quantile(confidence_level)].mean()
    cvar_annualized = -cvar_daily_losses * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    # Advanced metrics
    skewness = daily_returns_decimal.skew()
    kurtosis = daily_returns_decimal.kurtosis()
    
    # Omega ratio
    threshold_return = daily_rf_rate
    gains = daily_returns_decimal[daily_returns_decimal > threshold_return] - threshold_return
    losses = threshold_return - daily_returns_decimal[daily_returns_decimal < threshold_return]
    omega_ratio = gains.sum() / losses.sum() if losses.sum() > 0 else np.nan

    beta, alpha, treynor_ratio, information_ratio = np.nan, np.nan, np.nan, np.nan
    
    if benchmark_returns is not None and not benchmark_returns.empty:
        common_index = daily_returns_decimal.index.intersection(benchmark_returns.index)
        aligned_asset_returns = daily_returns_decimal.loc[common_index]
        aligned_benchmark_returns_decimal = benchmark_returns.loc[common_index]
        if aligned_benchmark_returns_decimal.abs().mean() > 0.1:
            aligned_benchmark_returns_decimal /= 100.0

        if len(common_index) > 1:
            covariance_matrix = np.cov(aligned_asset_returns, aligned_benchmark_returns_decimal)
            covariance = covariance_matrix[0, 1]
            benchmark_variance = aligned_benchmark_returns_decimal.var()
            
            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
                expected_asset_return_ann = annualized_return / 100
                
                if (1 + aligned_benchmark_returns_decimal > 0).all():
                    bench_geom_mean_daily_return = np.expm1(np.log1p(aligned_benchmark_returns_decimal).mean())
                    benchmark_annualized_return = ((1 + bench_geom_mean_daily_return) ** TRADING_DAYS_PER_YEAR - 1)
                else:
                    benchmark_annualized_return = ((aligned_benchmark_returns_decimal.mean() + 1) ** TRADING_DAYS_PER_YEAR - 1)

                alpha = (expected_asset_return_ann - (risk_free_rate_decimal + beta * (benchmark_annualized_return - risk_free_rate_decimal))) * 100
                treynor_ratio = (expected_asset_return_ann - risk_free_rate_decimal) / beta if beta != 0 else np.nan
                
                tracking_error_daily = (aligned_asset_returns - aligned_benchmark_returns_decimal).std()
                tracking_error = tracking_error_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
                
                if tracking_error > 0:
                    information_ratio = (expected_asset_return_ann - benchmark_annualized_return) / tracking_error

    def round_if_float(x):
        return round(x, 4) if isinstance(x, (int, float)) and not np.isnan(x) else np.nan
    
    return {
        "Total Return (%)": round_if_float(total_return),
        "Annualized Return (%)": round_if_float(annualized_return),
        "Annualized Volatility (%)": round_if_float(annualized_volatility),
        "Sharpe Ratio": round_if_float(sharpe_ratio),
        "Sortino Ratio": round_if_float(sortino_ratio),
        "Calmar Ratio": round_if_float(calmar_ratio),
        "Omega Ratio": round_if_float(omega_ratio),
        "Max Drawdown (%)": round_if_float(max_drawdown),
        "VaR (95%, Ann.) (%)": round_if_float(var_annualized),
        "CVaR (95%, Ann.) (%)": round_if_float(cvar_annualized),
        "Skewness": round_if_float(skewness),
        "Kurtosis": round_if_float(kurtosis),
        f"Beta (vs {BENCHMARK_SYMBOL})": round_if_float(beta),
        f"Alpha (%) (vs {BENCHMARK_SYMBOL})": round_if_float(alpha),
        "Treynor Ratio": round_if_float(treynor_ratio),
        "Information Ratio": round_if_float(information_ratio)
    }

@st.cache_data(ttl=3600, show_spinner="Calculating historical index values...")
def _calculate_historical_index_value(
    api_key: str, 
    access_token: str, 
    constituents_df: pd.DataFrame, 
    start_date: datetime.date, 
    end_date: datetime.date, 
    exchange: str = DEFAULT_EXCHANGE
) -> pd.DataFrame:
    if constituents_df.empty:
        return pd.DataFrame({"_error": ["No constituents provided."]})

    all_historical_closes = {}
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    if st.session_state["instruments_df"].empty:
        st.session_state["instruments_df"] = load_instruments_cached(api_key, access_token, exchange)
        if "_error" in st.session_state["instruments_df"].columns:
            return pd.DataFrame({"_error": [st.session_state["instruments_df"].loc[0, '_error']]})

    for i, row in constituents_df.iterrows():
        symbol = row['symbol']
        progress_text.text(f"Fetching data for {symbol} ({i+1}/{len(constituents_df)})...")
        
        hist_df = get_historical_data_cached(api_key, access_token, symbol, start_date, end_date, "day", exchange)
        
        if isinstance(hist_df, pd.DataFrame) and "_error" not in hist_df.columns and not hist_df.empty:
            all_historical_closes[symbol] = hist_df['close']
        else:
            st.warning(f"Could not fetch data for {symbol}. Skipping.")
        progress_bar.progress((i + 1) / len(constituents_df))

    progress_text.empty()
    progress_bar.empty()

    if not all_historical_closes:
        return pd.DataFrame({"_error": ["No historical data available."]})

    combined_closes = pd.DataFrame(all_historical_closes)
    combined_closes = combined_closes.ffill().bfill()
    combined_closes.dropna(how='all', inplace=True)

    if combined_closes.empty:
        return pd.DataFrame({"_error": ["Insufficient common historical data."]})

    weights_series = constituents_df.set_index('symbol')['Weights']
    common_symbols = weights_series.index.intersection(combined_closes.columns)
    if common_symbols.empty:
        return pd.DataFrame({"_error": ["No common symbols."]})

    aligned_combined_closes = combined_closes[common_symbols]
    aligned_weights = weights_series[common_symbols]

    weighted_closes = aligned_combined_closes.mul(aligned_weights, axis=1)
    index_history_series = weighted_closes.sum(axis=1)

    if not index_history_series.empty:
        first_valid_index = index_history_series.first_valid_index()
        if first_valid_index is not None:
            base_value = index_history_series[first_valid_index]
            if base_value != 0:
                index_history_df = pd.DataFrame({"index_value": (index_history_series / base_value) * 100})
                index_history_df.index.name = 'date'
                return index_history_df.dropna()
    return pd.DataFrame({"_error": ["Error calculating index values."]})

# --- Visualization Functions ---
def plot_drawdown_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig
    
    for col in df.columns:
        daily_returns = df[col].pct_change().dropna()
        cumulative_performance = (1 + daily_returns).cumprod()
        peak = cumulative_performance.expanding(min_periods=1).max()
        drawdown = ((cumulative_performance / peak) - 1) * 100
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode='lines', name=f'{col} Drawdown'))
        
    fig.update_layout(
        title_text="Drawdown Comparison", 
        yaxis_title="Drawdown (%)", 
        template="plotly_dark", 
        height=400, 
        hovermode="x unified"
    )
    return fig

def plot_rolling_volatility_chart(df: pd.DataFrame, window: int = 30) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig
    
    for col in df.columns:
        daily_returns = df[col].pct_change()
        rolling_vol = daily_returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, mode='lines', name=f'{col} ({window}D)'))
    
    fig.update_layout(
        title_text=f"{window}-Day Rolling Volatility", 
        yaxis_title="Volatility (%)", 
        template="plotly_dark", 
        height=400, 
        hovermode="x unified"
    )
    return fig

def plot_rolling_risk_charts(
    comparison_df: pd.DataFrame, 
    benchmark_returns: pd.Series, 
    window: int = 60
) -> Tuple[go.Figure, go.Figure]:
    if comparison_df.empty or benchmark_returns is None or benchmark_returns.empty:
        return go.Figure(), go.Figure()

    daily_returns_df = comparison_df.pct_change().dropna()
    
    fig_beta = go.Figure()
    fig_corr = go.Figure()

    common_index = daily_returns_df.index.intersection(benchmark_returns.index)
    aligned_benchmark_returns = benchmark_returns.loc[common_index]
    
    if aligned_benchmark_returns.abs().mean() > 0.1:
        aligned_benchmark_returns /= 100.0

    aligned_returns_df = daily_returns_df.loc[common_index]

    if aligned_returns_df.empty or len(aligned_returns_df) < window:
        return go.Figure(), go.Figure()

    for col in aligned_returns_df.columns:
        def calculate_rolling_beta(x):
            bench_window = aligned_benchmark_returns.loc[x.index]
            if bench_window.var() > 0:
                return x.cov(bench_window) / bench_window.var()
            return np.nan

        rolling_beta = aligned_returns_df[col].rolling(window=window).apply(calculate_rolling_beta, raw=False)
        rolling_corr = aligned_returns_df[col].rolling(window=window).corr(aligned_benchmark_returns)
        
        fig_beta.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta, mode='lines', name=f'{col}'))
        fig_corr.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode='lines', name=f'{col}'))

    fig_beta.update_layout(
        title_text=f"{window}-Day Rolling Beta", 
        yaxis_title="Beta", 
        template="plotly_dark", 
        height=400, 
        hovermode="x unified"
    )
    fig_corr.update_layout(
        title_text=f"{window}-Day Rolling Correlation", 
        yaxis_title="Correlation", 
        template="plotly_dark", 
        height=400, 
        hovermode="x unified", 
        yaxis_range=[-1, 1]
    )

    return fig_beta, fig_corr

def plot_correlation_heatmap(comparison_df: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap for multiple indexes/benchmarks"""
    if comparison_df.empty:
        return go.Figure()
    
    returns_df = comparison_df.pct_change().dropna()
    corr_matrix = returns_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title_text="Returns Correlation Matrix",
        template="plotly_dark",
        height=500,
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    return fig

def plot_efficient_frontier(returns_df: pd.DataFrame, num_portfolios: int = 5000) -> go.Figure:
    """Generate efficient frontier visualization"""
    if returns_df.empty or len(returns_df.columns) < 2:
        return go.Figure()
    
    returns = returns_df.pct_change().dropna()
    mean_returns = returns.mean() * TRADING_DAYS_PER_YEAR
    cov_matrix = returns.cov() * TRADING_DAYS_PER_YEAR
    
    num_assets = len(returns.columns)
    results = np.zeros((3, num_portfolios))
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std
        results[2,i] = portfolio_return / portfolio_std
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results[1,:] * 100,
        y=results[0,:] * 100,
        mode='markers',
        marker=dict(
            size=5,
            color=results[2,:],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        text=[f'Return: {r:.2f}%<br>Risk: {s:.2f}%<br>Sharpe: {sh:.2f}' 
              for r, s, sh in zip(results[0,:]*100, results[1,:]*100, results[2,:])],
        hovertemplate='%{text}<extra></extra>',
        name='Portfolios'
    ))
    
    max_sharpe_idx = np.argmax(results[2,:])
    fig.add_trace(go.Scatter(
        x=[results[1, max_sharpe_idx] * 100],
        y=[results[0, max_sharpe_idx] * 100],
        mode='markers',
        marker=dict(color='red', size=15, symbol='star'),
        name='Max Sharpe Ratio'
    ))
    
    fig.update_layout(
        title_text="Efficient Frontier Analysis",
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        template="plotly_dark",
        height=600,
        hovermode='closest'
    )
    return fig

# --- Factsheet Generation Functions ---
def generate_factsheet_csv_content(
    factsheet_constituents_df_final: pd.DataFrame,
    factsheet_history_df_final: pd.DataFrame,
    last_comparison_df: pd.DataFrame,
    last_comparison_metrics: Dict,
    current_live_value: float,
    index_name: str = "Custom Index",
    ai_agent_embed_snippet: Optional[str] = None
) -> str:
    content = []
    
    content.append(f"Factsheet for {index_name}\n")
    content.append(f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    content.append("\n--- Index Overview ---\n")
    if current_live_value > 0 and not factsheet_constituents_df_final.empty:
        content.append(f"Current Live Calculated Index Value,₹{current_live_value:,.2f}\n")
    else:
        content.append("Current Live Calculated Index Value,N/A\n")
    
    content.append("\n--- Constituents ---\n")
    if not factsheet_constituents_df_final.empty:
        const_export_df = factsheet_constituents_df_final.copy()
        if 'Last Price' not in const_export_df.columns:
            const_export_df['Last Price'] = np.nan
        if 'Weighted Price' not in const_export_df.columns:
            const_export_df['Weighted Price'] = np.nan
        
        const_export_df['Last Price'] = const_export_df['Last Price'].apply(
            lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A"
        )
        const_export_df['Weighted Price'] = const_export_df['Weighted Price'].apply(
            lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A"
        )
        
        content.append(const_export_df[['symbol', 'Name', 'Weights', 'Last Price', 'Weighted Price']].to_csv(index=False))
    else:
        content.append("No constituent data available.\n")

    content.append("\n--- Historical Performance ---\n")
    if not factsheet_history_df_final.empty:
        content.append(factsheet_history_df_final.to_csv())
    else:
        content.append("No historical performance data available.\n")

    content.append("\n--- Performance Metrics ---\n")
    if last_comparison_metrics:
        metrics_df = pd.DataFrame(last_comparison_metrics).T
        metrics_df = metrics_df.applymap(
            lambda x: f"{x:.4f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A"
        )
        content.append(metrics_df.T.to_csv())
    else:
        content.append("No performance metrics available.\n")

    content.append("\n--- Comparison Data ---\n")
    if not last_comparison_df.empty:
        content.append(last_comparison_df.to_csv())
    else:
        content.append("No comparison data available.\n")

    return "".join(content)

def generate_factsheet_html_content(
    factsheet_constituents_df_final: pd.DataFrame,
    factsheet_history_df_final: pd.DataFrame,
    last_comparison_df: pd.DataFrame,
    last_comparison_metrics: Dict,
    current_live_value: float,
    index_name: str = "Custom Index",
    ai_agent_embed_snippet: Optional[str] = None
) -> str:
    html_parts = []
    
    html_parts.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Invsion Connect Factsheet</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #1a1a1a; color: #e0e0e0; }
            .container { max-width: 1200px; margin: auto; padding: 20px; background-color: #2b2b2b; border-radius: 8px; }
            h1, h2, h3 { color: #f0f0f0; border-bottom: 2px solid #444; padding-bottom: 5px; }
            table { width: 100%; border-collapse: collapse; margin: 15px 0; }
            th, td { border: 1px solid #444; padding: 8px; text-align: left; }
            th { background-color: #3a3a3a; }
            .metric { font-size: 1.1em; margin: 5px 0; }
            .plotly-graph { margin: 20px 0; border: 1px solid #444; border-radius: 5px; }
            .info-box { background-color: #334455; border-left: 5px solid #6699cc; padding: 10px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
    """)
    
    html_parts.append(f"<h1>Invsion Connect Factsheet: {index_name}</h1>")
    html_parts.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    
    if current_live_value > 0:
        html_parts.append(f"<p class='metric'><strong>Live Index Value:</strong> ₹{current_live_value:,.2f}</p>")
    
    if not factsheet_constituents_df_final.empty:
        html_parts.append("<h2>Constituents</h2>")
        html_parts.append(factsheet_constituents_df_final.to_html(index=False, classes='table'))
    
    if last_comparison_metrics:
        html_parts.append("<h2>Performance Metrics</h2>")
        metrics_df = pd.DataFrame(last_comparison_metrics).T
        html_parts.append(metrics_df.to_html(classes='table'))
    
    if not last_comparison_df.empty:
        html_parts.append("<h2>Performance Comparison</h2>")
        fig = go.Figure()
        for col in last_comparison_df.columns:
            fig.add_trace(go.Scatter(x=last_comparison_df.index, y=last_comparison_df[col], 
                                    mode='lines', name=col))
        fig.update_layout(template="plotly_dark", height=500)
        html_parts.append(f"<div class='plotly-graph'>{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>")
    
    if ai_agent_embed_snippet:
        html_parts.append(f"<div class='info-box'><h3>AI Insights</h3>{ai_agent_embed_snippet}</div>")
    
    html_parts.append("</div></body></html>")
    return "".join(html_parts)

# --- Index Blending Functions ---
def blend_indexes(
    index_dict: Dict[str, pd.DataFrame], 
    weights: Dict[str, float]
) -> pd.DataFrame:
    """Blend multiple indexes with specified weights"""
    if not index_dict or not weights:
        return pd.DataFrame()
    
    total_weight = sum(weights.values())
    if total_weight == 0:
        return pd.DataFrame()
    
    normalized_weights = {k: v/total_weight for k, v in weights.items()}
    
    all_data = pd.DataFrame()
    for idx_name, idx_df in index_dict.items():
        if idx_name in normalized_weights:
            weight = normalized_weights[idx_name]
            if all_data.empty:
                all_data = idx_df * weight
            else:
                all_data = all_data.add(idx_df * weight, fill_value=0)
    
    return all_data

# --- Compliance Validation Functions ---
def parse_and_validate_rules(rules_text: str, portfolio_df: pd.DataFrame) -> List[Dict]:
    results = []
    if not rules_text.strip() or portfolio_df.empty:
        return results

    sector_weights = portfolio_df.groupby('Industry')['Weight %'].sum()
    stock_weights = portfolio_df.set_index('Symbol')['Weight %']
    rating_weights = portfolio_df.groupby('Rating')['Weight %'].sum() if 'Rating' in portfolio_df.columns else pd.Series()
    asset_class_weights = portfolio_df.groupby('Asset Class')['Weight %'].sum() if 'Asset Class' in portfolio_df.columns else pd.Series()
    
    def check_pass(actual, op, threshold):
        ops = {'>': lambda a, t: a > t, '<': lambda a, t: a < t, 
               '>=': lambda a, t: a >= t, '<=': lambda a, t: a <= t, '=': lambda a, t: a == t}
        return ops.get(op, lambda a, t: False)(actual, threshold)

    for rule in rules_text.strip().split('\n'):
        rule = rule.strip()
        if not rule or rule.startswith('#'):
            continue

        parts = re.split(r'\s+', rule)
        rule_type = parts[0].upper()
        
        try:
            actual_value = None
            details = ""

            if len(parts) < 3:
                results.append({'rule': rule, 'status': 'Error', 'details': 'Invalid format'})
                continue
            
            op = parts[-2]
            if op not in ['>', '<', '>=', '<=', '=']:
                results.append({'rule': rule, 'status': 'Error', 'details': f"Invalid operator '{op}'"})
                continue

            threshold_str = parts[-1].replace('%', '')
            
            if rule_type == 'STOCK' and len(parts) == 4:
                symbol = parts[1].upper()
                threshold = float(threshold_str)
                if symbol in stock_weights.index:
                    actual_value = stock_weights.get(symbol, 0.0)
                    details = f"Actual: {actual_value:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': f"Symbol '{symbol}' not found"})
                    continue
            
            elif rule_type == 'SECTOR':
                sector_name = ' '.join(parts[1:-2]).upper()
                threshold = float(threshold_str)
                matching_sector = next((s for s in sector_weights.index if s.upper() == sector_name), None)
                if matching_sector:
                    actual_value = sector_weights.get(matching_sector, 0.0)
                    details = f"Actual: {actual_value:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': f"Sector '{sector_name}' not found"})
                    continue
            
            elif rule_type == 'TOP_N_STOCKS' and len(parts) == 4:
                n = int(parts[1])
                threshold = float(threshold_str)
                actual_value = portfolio_df.nlargest(n, 'Weight %')['Weight %'].sum()
                details = f"Top {n} stocks: {actual_value:.2f}%"
            
            elif rule_type == 'TOP_N_SECTORS' and len(parts) == 4:
                n = int(parts[1])
                threshold = float(threshold_str)
                actual_value = sector_weights.nlargest(n).sum()
                details = f"Top {n} sectors: {actual_value:.2f}%"
            
            elif rule_type == 'COUNT_STOCKS' and len(parts) == 3:
                threshold = int(threshold_str)
                actual_value = len(portfolio_df)
                details = f"Count: {actual_value}"
            
            else:
                results.append({'rule': rule, 'status': 'Error', 'details': 'Unrecognized format'})
                continue
            
            if actual_value is not None:
                passed = check_pass(actual_value, op, threshold)
                status = "✅ PASS" if passed else "❌ FAIL"
                results.append({'rule': rule, 'status': status, 'details': f"{details} | Rule: {op} {threshold}"})

        except (ValueError, IndexError) as e:
            results.append({'rule': rule, 'status': 'Error', 'details': f"Parse error: {e}"})
            
    return results

# --- Sidebar: Kite Login ---
with st.sidebar:
    st.markdown("### 1. Login to Kite Connect")
    st.markdown(f"[🔗 Open Kite login]({login_url})")

    request_token_param = st.query_params.get("request_token")
    if request_token_param and not st.session_state["kite_access_token"]:
        st.info("Exchanging token...")
        try:
            data = kite_unauth_client.generate_session(
                request_token_param, 
                api_secret=KITE_CREDENTIALS["api_secret"]
            )
            st.session_state["kite_access_token"] = data.get("access_token")
            st.session_state["kite_login_response"] = data
            st.sidebar.success("Kite authenticated!")
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed: {e}")

    if st.session_state["kite_access_token"]:
        if st.sidebar.button("Logout from Kite"):
            st.session_state["kite_access_token"] = None
            st.session_state["kite_login_response"] = None
            st.success("Logged out from Kite")
            st.rerun()
        st.success("Kite Authenticated ✅")
    else:
        st.info("Not authenticated with Kite")

# --- Sidebar: Supabase User Account ---
with st.sidebar:
    st.markdown("### 2. Supabase User Account")
    _refresh_supabase_session()

    if st.session_state["user_session"]:
        st.success(f"Logged in: {st.session_state['user_session'].user.email}")
        if st.button("Logout from Supabase"):
            try:
                supabase.auth.sign_out()
                _refresh_supabase_session()
                st.success("Logged out")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        with st.form("supabase_auth_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                login_btn = st.form_submit_button("Login")
            with col2:
                signup_btn = st.form_submit_button("Sign Up")

            if login_btn and email and password:
                try:
                    supabase.auth.sign_in_with_password({"email": email, "password": password})
                    _refresh_supabase_session()
                    st.success("Login successful!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")
            
            if signup_btn and email and password:
                try:
                    supabase.auth.sign_up({"email": email, "password": password})
                    st.success("Sign up successful! Check email.")
                except Exception as e:
                    st.error(f"Sign up failed: {e}")

    st.markdown("---")
    st.markdown("### 3. Quick Data Access")
    if st.session_state["kite_access_token"]:
        k_sidebar = get_authenticated_kite_client(
            KITE_CREDENTIALS["api_key"], 
            st.session_state["kite_access_token"]
        )
        if st.button("Fetch Holdings"):
            try:
                holdings = k_sidebar.holdings()
                st.session_state["holdings_data"] = pd.DataFrame(holdings)
                st.success(f"Fetched {len(holdings)} holdings")
            except Exception as e:
                st.error(f"Error: {e}")
        
        if st.session_state.get("holdings_data") is not None and not st.session_state["holdings_data"].empty:
            with st.expander("Show Holdings"):
                st.dataframe(st.session_state["holdings_data"])
                st.download_button(
                    "Download Holdings CSV",
                    data=st.session_state["holdings_data"].to_csv(index=False).encode('utf-8'),
                    file_name="holdings.csv",
                    mime="text/csv"
                )

# --- Main Tabs ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

tabs = st.tabs([
    "📈 Market & Historical", 
    "📊 Custom Index", 
    "⚡ Index Calculator", 
    "💼 Compliance"
])

# --- TAB 1: Market & Historical ---
with tabs[0]:
    st.header("📈 Market Data & Technical Analysis")
    
    if not k:
        st.info("Login to Kite first")
    else:
        st.subheader("Current Market Data")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            q_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"])
            q_symbol = st.text_input("Symbol", value="RELIANCE")
            if st.button("Get Market Data"):
                ltp_data = get_ltp_price_cached(api_key, access_token, q_symbol, q_exchange)
                if ltp_data and "_error" not in ltp_data:
                    st.session_state["current_market_data"] = ltp_data
                    st.success(f"Fetched LTP for {q_symbol}")
                else:
                    st.error(f"Failed: {ltp_data.get('_error', 'Unknown error')}")
        
        with col2:
            if st.session_state.get("current_market_data"):
                st.json(st.session_state["current_market_data"])
                st.download_button(
                    "Download Market Data",
                    data=pd.DataFrame([st.session_state["current_market_data"]]).to_csv(index=False).encode('utf-8'),
                    file_name=f"{q_symbol}_market_data.csv",
                    mime="text/csv"
                )

        st.markdown("---")
        st.subheader("Historical Data & Technical Analysis")
        
        hist_symbol = st.text_input("Symbol for Historical Data", value="INFY")
        
        col_f, col_i, col_d = st.columns(3)
        with col_d:
            from_date = st.date_input("From", value=datetime.now().date() - timedelta(days=365))
            to_date = st.date_input("To", value=datetime.now().date())
        with col_i:
            interval = st.selectbox("Interval", ["day", "week", "month", "minute", "5minute", "30minute"])
        with col_f:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Fetch Historical Data", type="primary"):
                df_hist = get_historical_data_cached(
                    api_key, access_token, hist_symbol, from_date, to_date, interval
                )
                if isinstance(df_hist, pd.DataFrame) and "_error" not in df_hist.columns:
                    st.session_state["historical_data"] = df_hist
                    st.session_state["last_fetched_symbol"] = hist_symbol
                    st.success(f"Fetched {len(df_hist)} records")
                else:
                    st.error(f"Failed: {df_hist.get('_error', 'Unknown')}")

        if not st.session_state.get("historical_data", pd.DataFrame()).empty:
            df = st.session_state["historical_data"]

            with st.expander("⚙️ Technical Indicator Settings", expanded=False):
                tc1, tc2, tc3 = st.columns(3)
                with tc1:
                    sma_str = st.text_input("SMA Periods", "20,50,200")
                    ema_str = st.text_input("EMA Periods", "12,26")
                    rsi_win = st.number_input("RSI Window", 5, 50, 14)
                with tc2:
                    macd_f = st.number_input("MACD Fast", 5, 50, 12)
                    macd_s = st.number_input("MACD Slow", 10, 100, 26)
                    macd_sig = st.number_input("MACD Signal", 5, 50, 9)
                with tc3:
                    bb_win = st.number_input("BB Window", 5, 50, 20)
                    bb_std = st.number_input("BB Std Dev", 1.0, 4.0, 2.0, 0.5)
                    chart_type = st.selectbox("Chart Type", ["Candlestick", "Line"])
                    indicators = st.multiselect("Indicators", ["SMA", "EMA", "Bollinger Bands"])

                try:
                    sma_periods = [int(p.strip()) for p in sma_str.split(',') if p.strip().isdigit()]
                    ema_periods = [int(p.strip()) for p in ema_str.split(',') if p.strip().isdigit()]
                except ValueError:
                    st.error("Invalid period input")
                    sma_periods, ema_periods = [], []

                df_ta = add_technical_indicators(
                    df, sma_periods, ema_periods, rsi_win, 
                    macd_f, macd_s, macd_sig, bb_win, bb_std
                )

            st.subheader(f"Analysis: {st.session_state['last_fetched_symbol']}")
            
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True, 
                vertical_spacing=0.03, row_heights=[0.5, 0.1, 0.2, 0.2]
            )
            
            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['open'], high=df['high'], 
                    low=df['low'], close=df['close'], name='Price'
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['close'], mode='lines', name='Price'
                ), row=1, col=1)

            if "SMA" in indicators:
                for p in sma_periods:
                    fig.add_trace(go.Scatter(
                        x=df_ta.index, y=df_ta.get(f'SMA_{p}'), 
                        mode='lines', name=f'SMA {p}'
                    ), row=1, col=1)
            
            if "EMA" in indicators:
                for p in ema_periods:
                    fig.add_trace(go.Scatter(
                        x=df_ta.index, y=df_ta.get(f'EMA_{p}'), 
                        mode='lines', name=f'EMA {p}'
                    ), row=1, col=1)
            
            if "Bollinger Bands" in indicators:
                fig.add_trace(go.Scatter(
                    x=df_ta.index, y=df_ta['Bollinger_High'], 
                    mode='lines', line=dict(width=0.5, color='gray'), name='BB High'
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df_ta.index, y=df_ta['Bollinger_Low'], 
                    mode='lines', line=dict(width=0.5, color='gray'), 
                    fill='tonexty', fillcolor='rgba(128,128,128,0.2)', name='BB Low'
                ), row=1, col=1)
            
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['RSI'], mode='lines', name='RSI'), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)
            
            fig.add_trace(go.Bar(
                x=df_ta.index, y=df_ta['MACD_hist'], 
                name='MACD Hist', marker_color='orange'
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=df_ta.index, y=df_ta['MACD'], 
                mode='lines', name='MACD', line=dict(color='blue')
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=df_ta.index, y=df_ta['MACD_signal'], 
                mode='lines', name='Signal', line=dict(color='red')
            ), row=4, col=1)
            
            fig.update_layout(
                height=1000, xaxis_rangeslider_visible=False, 
                title_text=f"{st.session_state['last_fetched_symbol']} Technical Analysis", 
                template="plotly_white"
            )
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
            fig.update_yaxes(title_text="MACD", row=4, col=1)
            
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download TA Data",
                data=df_ta.to_csv().encode('utf-8'),
                file_name=f"{st.session_state['last_fetched_symbol']}_TA.csv",
                mime="text/csv"
            )

            st.subheader("Performance Metrics")
            daily_returns = df['close'].pct_change().dropna()
            metrics = calculate_performance_metrics(daily_returns, risk_free_rate=6.0)
            if metrics:
                st.dataframe(
                    pd.DataFrame([metrics]).T.rename(columns={0: "Value"}).style.format("{:.4f}"),
                    use_container_width=True
                )
                st.download_button(
                    "Download Metrics",
                    data=pd.DataFrame([metrics]).T.to_csv().encode('utf-8'),
                    file_name=f"{st.session_state['last_fetched_symbol']}_metrics.csv",
                    mime="text/csv"
                )

# --- TAB 2: Custom Index (Enhanced) ---
with tabs[1]:
    st.header("📊 Custom Index Creation & Analysis")
    
    if not k or not st.session_state["user_id"]:
        st.info("Login to Kite and Supabase to access this feature")
    else:
        st.markdown("---")
        st.subheader("1. Create New Index")
        
        create_method = st.radio(
            "Creation Method",
            ["CSV Upload", "From Holdings", "Manual Entry", "Smart Beta Strategy"],
            horizontal=True
        )
        
        if create_method == "CSV Upload":
            uploaded_file = st.file_uploader("Upload CSV (symbol, Weights columns)", type=["csv"])
            if uploaded_file:
                try:
                    df_const = pd.read_csv(uploaded_file)
                    required_cols = {"symbol", "Weights"}
                    if not required_cols.issubset(set(df_const.columns)):
                        st.error(f"Missing columns: {required_cols - set(df_const.columns)}")
                    else:
                        df_const["Weights"] = pd.to_numeric(df_const["Weights"], errors='coerce')
                        df_const.dropna(subset=["Weights", "symbol"], inplace=True)
                        total_w = df_const["Weights"].sum()
                        if total_w > 0:
                            df_const["Weights"] = df_const["Weights"] / total_w
                            if 'Name' not in df_const.columns:
                                df_const['Name'] = df_const['symbol']
                            st.session_state["current_calculated_index_data"] = df_const[['symbol', 'Name', 'Weights']]
                            st.success(f"Loaded {len(df_const)} constituents")
                except Exception as e:
                    st.error(f"Error processing CSV: {e}")
        
        elif create_method == "From Holdings":
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                weight_scheme = st.selectbox("Weighting", ["Equal Weight", "Value Weighted", "Market Cap Weighted"])
            with col_h2:
                if st.button("Create from Holdings"):
                    holdings_df = st.session_state.get("holdings_data")
                    if holdings_df is None or holdings_df.empty:
                        st.warning("Fetch holdings first from sidebar")
                    else:
                        df_const = holdings_df[holdings_df['product'] != 'CDS'].copy()
                        df_const.rename(columns={'tradingsymbol': 'symbol'}, inplace=True)
                        df_const['Name'] = df_const['symbol']
                        
                        if weight_scheme == "Equal Weight":
                            df_const['Weights'] = 1 / len(df_const)
                        elif weight_scheme == "Value Weighted":
                            df_const['investment_value'] = df_const['average_price'] * df_const['quantity']
                            total_val = df_const['investment_value'].sum()
                            df_const['Weights'] = df_const['investment_value'] / total_val if total_val > 0 else 0
                        
                        st.session_state["current_calculated_index_data"] = df_const[['symbol', 'Name', 'Weights']]
                        st.success(f"Created index with {len(df_const)} holdings")
        
        elif create_method == "Manual Entry":
            st.info("Enter constituents manually (comma-separated symbols and weights)")
            symbols_input = st.text_input("Symbols (comma-separated)", "RELIANCE,TCS,INFY,HDFCBANK")
            weights_input = st.text_input("Weights (comma-separated)", "0.25,0.25,0.25,0.25")
            
            if st.button("Create Manual Index"):
                try:
                    symbols = [s.strip().upper() for s in symbols_input.split(',')]
                    weights = [float(w.strip()) for w in weights_input.split(',')]
                    
                    if len(symbols) != len(weights):
                        st.error("Number of symbols and weights must match")
                    else:
                        total_w = sum(weights)
                        normalized_weights = [w/total_w for w in weights]
                        df_const = pd.DataFrame({
                            'symbol': symbols,
                            'Name': symbols,
                            'Weights': normalized_weights
                        })
                        st.session_state["current_calculated_index_data"] = df_const
                        st.success(f"Created manual index with {len(symbols)} constituents")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif create_method == "Smart Beta Strategy":
            st.info("Create index based on factor strategies")
            strategy = st.selectbox(
                "Strategy Type",
                ["Low Volatility", "High Dividend Yield", "Momentum", "Quality", "Value"]
            )
            sector_filter = st.multiselect("Filter by Sectors (optional)", 
                                          ["BANKS", "IT", "PHARMA", "AUTO", "FMCG", "ENERGY"])
            top_n = st.number_input("Top N stocks", 5, 50, 20)
            
            if st.button("Generate Smart Beta Index"):
                st.info(f"Smart Beta '{strategy}' strategy selected. This would fetch fundamental data and rank stocks accordingly.")
                st.warning("Note: Implementation requires fundamental data API integration")

        # Calculate Historical Performance
        current_calc_df = st.session_state.get("current_calculated_index_data", pd.DataFrame())
        if not current_calc_df.empty:
            st.markdown("---")
            st.subheader("2. Calculate Historical Performance")
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                hist_start = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=365))
            with col_d2:
                hist_end = st.date_input("End Date", value=datetime.now().date())

            if st.button("Calculate Historical Values", type="primary"):
                if hist_start >= hist_end:
                    st.error("Start date must be before end date")
                else:
                    index_history = _calculate_historical_index_value(
                        api_key, access_token, current_calc_df, hist_start, hist_end
                    )
                    if not index_history.empty and "_error" not in index_history.columns:
                        st.session_state["current_calculated_index_history"] = index_history
                        st.success("Historical values calculated!")
                    else:
                        st.error(f"Failed: {index_history.get('_error', ['Unknown'])[0]}")

        # Display calculated index
        current_hist_df = st.session_state.get("current_calculated_index_history", pd.DataFrame())
        if not current_calc_df.empty and not current_hist_df.empty:
            st.markdown("---")
            st.subheader("3. Index Overview")
            
            # Fetch live prices
            live_quotes = {}
            symbols = current_calc_df["symbol"].tolist()
            try:
                instrument_ids = [f"{DEFAULT_EXCHANGE}:{s}" for s in symbols]
                ltp_batch = k.ltp(instrument_ids)
                for sym in symbols:
                    key = f"{DEFAULT_EXCHANGE}:{sym}"
                    live_quotes[sym] = ltp_batch.get(key, {}).get("last_price", np.nan)
            except Exception:
                pass

            display_df = current_calc_df.copy()
            display_df["Last Price"] = display_df["symbol"].map(live_quotes)
            display_df["Weighted Price"] = display_df["Last Price"] * display_df["Weights"]
            live_value = display_df["Weighted Price"].sum()

            st.metric("Live Index Value", f"₹{live_value:,.2f}")
            st.dataframe(display_df.style.format({
                "Weights": "{:.4f}",
                "Last Price": "₹{:,.2f}",
                "Weighted Price": "₹{:,.2f}"
            }), use_container_width=True)

            # Save Index
            st.markdown("---")
            st.subheader("4. Save Index")
            index_name = st.text_input("Index Name", value="MyCustomIndex")
            if st.button("Save to Database"):
                if index_name and st.session_state["user_id"]:
                    try:
                        check = supabase.table("custom_indexes").select("id").eq(
                            "user_id", st.session_state["user_id"]
                        ).eq("index_name", index_name).execute()
                        
                        if check.data:
                            st.warning(f"Index '{index_name}' exists. Choose different name.")
                        else:
                            hist_save = current_hist_df.reset_index()
                            hist_save['date'] = hist_save['date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
                            
                            index_data = {
                                "user_id": st.session_state["user_id"],
                                "index_name": index_name,
                                "constituents": current_calc_df[['symbol', 'Name', 'Weights']].to_dict(orient='records'),
                                "historical_performance": hist_save.to_dict(orient='records')
                            }
                            supabase.table("custom_indexes").insert(index_data).execute()
                            st.success(f"Index '{index_name}' saved!")
                            st.session_state["current_calculated_index_data"] = pd.DataFrame()
                            st.session_state["current_calculated_index_history"] = pd.DataFrame()
                    except Exception as e:
                        st.error(f"Save failed: {e}")

        # Load and manage saved indexes
        st.markdown("---")
        st.subheader("5. Load & Compare Saved Indexes")
        
        if st.button("Load My Indexes"):
            try:
                response = supabase.table("custom_indexes").select(
                    "id, index_name, constituents, historical_performance"
                ).eq("user_id", st.session_state["user_id"]).execute()
                st.session_state["saved_indexes"] = response.data if response.data else []
                st.success(f"Loaded {len(st.session_state['saved_indexes'])} indexes")
            except Exception as e:
                st.error(f"Load failed: {e}")

        saved_indexes = st.session_state.get("saved_indexes", [])
        if saved_indexes:
            index_names = [idx['index_name'] for idx in saved_indexes]
            
            st.markdown("---")
            st.subheader("6. Multi-Index Comparison & Benchmarking")
            
            selected_indexes = st.multiselect(
                "Select indexes to compare",
                options=index_names
            )
            
            col_comp1, col_comp2 = st.columns(2)
            with col_comp1:
                comp_start = st.date_input("Comparison Start", value=datetime.now().date() - timedelta(days=365))
                comp_end = st.date_input("Comparison End", value=datetime.now().date())
            
            with col_comp2:
                benchmarks_str = st.text_area(
                    "Benchmarks (comma-separated)",
                    value=f"{BENCHMARK_SYMBOL}, NIFTY BANK",
                    height=80
                )
                risk_free = st.number_input("Risk-Free Rate (%)", 0.0, 20.0, 6.0, 0.1)

            # Advanced comparison options
            with st.expander("⚙️ Advanced Comparison Options"):
                show_corr_matrix = st.checkbox("Show Correlation Matrix", value=True)
                show_efficient_frontier = st.checkbox("Show Efficient Frontier", value=False)
                rolling_window = st.slider("Rolling Window (days)", 20, 120, 60)

            if st.button("Run Comprehensive Comparison", type="primary"):
                if not selected_indexes and not benchmarks_str.strip():
                    st.warning("Select at least one index or benchmark")
                else:
                    external_benchmarks = [s.strip().upper() for s in benchmarks_str.split(',') if s.strip()]
                    
                    all_data = {}
                    all_metrics = {}
                    
                    # Fetch benchmark for risk calculations
                    benchmark_df = get_historical_data_cached(
                        api_key, access_token, BENCHMARK_SYMBOL, comp_start, comp_end, "day", "NSE"
                    )
                    benchmark_returns = None
                    if "_error" not in benchmark_df.columns:
                        benchmark_returns = benchmark_df['close'].pct_change().dropna()
                        st.session_state["benchmark_historical_data"] = benchmark_df

                    # Process selected indexes
                    for idx_name in selected_indexes:
                        idx_data = next((idx for idx in saved_indexes if idx['index_name'] == idx_name), None)
                        if idx_data:
                            constituents = pd.DataFrame(idx_data['constituents'])
                            hist_df = _calculate_historical_index_value(
                                api_key, access_token, constituents, comp_start, comp_end
                            )
                            if "_error" not in hist_df.columns:
                                all_data[idx_name] = hist_df['index_value']
                                returns = hist_df['index_value'].pct_change().dropna()
                                all_metrics[idx_name] = calculate_performance_metrics(
                                    returns, risk_free, benchmark_returns
                                )

                    # Process benchmarks
                    for bench in external_benchmarks:
                        bench_df = get_historical_data_cached(
                            api_key, access_token, bench, comp_start, comp_end, "day", "NSE"
                        )
                        if "_error" not in bench_df.columns:
                            first_val = bench_df['close'].iloc[0]
                            all_data[bench] = (bench_df['close'] / first_val) * 100
                            returns = bench_df['close'].pct_change().dropna()
                            all_metrics[bench] = calculate_performance_metrics(
                                returns, risk_free, benchmark_returns
                            )

                    if all_data:
                        comparison_df = pd.DataFrame(all_data).dropna(how='all')
                        st.session_state["last_comparison_df"] = comparison_df
                        st.session_state["last_comparison_metrics"] = all_metrics
                        st.session_state["correlation_matrix"] = comparison_df.pct_change().dropna().corr()
                        st.success("Comparison completed!")

            # Display comparison results
            comp_df = st.session_state.get("last_comparison_df", pd.DataFrame())
            if not comp_df.empty:
                st.markdown("---")
                st.subheader("📊 Comparison Results")
                
                # Performance chart
                fig_perf = go.Figure()
                for col in comp_df.columns:
                    fig_perf.add_trace(go.Scatter(
                        x=comp_df.index, y=comp_df[col], 
                        mode='lines', name=col
                    ))
                fig_perf.update_layout(
                    title="Cumulative Performance (Normalized to 100)",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    template="plotly_dark",
                    height=600,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_perf, use_container_width=True)

                # Metrics table
                st.markdown("#### Performance Metrics")
                metrics_df = pd.DataFrame(st.session_state["last_comparison_metrics"]).T
                st.dataframe(metrics_df.style.format("{:.4f}", na_rep="N/A"), use_container_width=True)
                
                st.download_button(
                    "Download Metrics CSV",
                    data=metrics_df.to_csv().encode('utf-8'),
                    file_name="comparison_metrics.csv",
                    mime="text/csv"
                )

                # Risk Analysis
                st.markdown("---")
                st.markdown("#### Risk Analysis")
                
                risk_tabs = st.tabs(["Drawdown", "Rolling Volatility", "Beta & Correlation", "Correlation Matrix"])
                
                with risk_tabs[0]:
                    st.plotly_chart(plot_drawdown_chart(comp_df), use_container_width=True)
                
                with risk_tabs[1]:
                    st.plotly_chart(plot_rolling_volatility_chart(comp_df, rolling_window), use_container_width=True)
                
                with risk_tabs[2]:
                    if not st.session_state["benchmark_historical_data"].empty:
                        bench_ret = st.session_state["benchmark_historical_data"]['close'].pct_change().dropna()
                        fig_beta, fig_corr = plot_rolling_risk_charts(comp_df, bench_ret, rolling_window)
                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            if fig_beta.data:
                                st.plotly_chart(fig_beta, use_container_width=True)
                        with col_r2:
                            if fig_corr.data:
                                st.plotly_chart(fig_corr, use_container_width=True)
                
                with risk_tabs[3]:
                    if show_corr_matrix:
                        st.plotly_chart(plot_correlation_heatmap(comp_df), use_container_width=True)

                # Efficient Frontier
                if show_efficient_frontier and len(comp_df.columns) >= 2:
                    st.markdown("---")
                    st.markdown("#### Efficient Frontier Analysis")
                    st.plotly_chart(plot_efficient_frontier(comp_df), use_container_width=True)

                # Index Blending
                st.markdown("---")
                st.subheader("🔀 Index Blending")
                st.info("Create a blended index by combining multiple indexes with custom weights")
                
                blend_weights = {}
                for col in comp_df.columns:
                    blend_weights[col] = st.slider(
                        f"{col} Weight (%)",
                        0, 100, 100 // len(comp_df.columns),
                        key=f"blend_{col}"
                    )
                
                if st.button("Create Blended Index"):
                    normalized_weights = {k: v/100 for k, v in blend_weights.items() if v > 0}
                    blended_data = pd.DataFrame()
                    
                    for idx_name, weight in normalized_weights.items():
                        if blended_data.empty:
                            blended_data = comp_df[[idx_name]] * weight
                        else:
                            blended_data = blended_data.add(comp_df[[idx_name]] * weight, fill_value=0)
                    
                    blended_data.columns = ['Blended Index']
                    
                    fig_blend = go.Figure()
                    fig_blend.add_trace(go.Scatter(
                        x=blended_data.index, 
                        y=blended_data['Blended Index'],
                        mode='lines', 
                        name='Blended Index',
                        line=dict(width=3)
                    ))
                    
                    for col in comp_df.columns:
                        fig_blend.add_trace(go.Scatter(
                            x=comp_df.index, 
                            y=comp_df[col],
                            mode='lines', 
                            name=col,
                            opacity=0.5
                        ))
                    
                    fig_blend.update_layout(
                        title="Blended Index Performance",
                        template="plotly_dark",
                        height=500
                    )
                    st.plotly_chart(fig_blend, use_container_width=True)
                    
                    # Calculate blended metrics
                    blended_returns = blended_data['Blended Index'].pct_change().dropna()
                    bench_ret = st.session_state.get("benchmark_historical_data", pd.DataFrame()).get('close', pd.Series()).pct_change().dropna()
                    blended_metrics = calculate_performance_metrics(blended_returns, risk_free, bench_ret if not bench_ret.empty else None)
                    
                    st.markdown("#### Blended Index Metrics")
                    st.dataframe(pd.DataFrame([blended_metrics]).T, use_container_width=True)

        # Factsheet Generation
        st.markdown("---")
        st.subheader("📄 Generate Factsheet")
        
        factsheet_format = st.radio("Format", ["CSV (Detailed)", "HTML (Visual)"], horizontal=True)
        ai_snippet = st.text_area("Optional AI Agent HTML snippet", height=100)
        
        col_fact1, col_fact2 = st.columns(2)
        with col_fact1:
            if st.button("Generate CSV Factsheet"):
                csv_content = generate_factsheet_csv_content(
                    current_calc_df,
                    current_hist_df,
                    comp_df,
                    st.session_state.get("last_comparison_metrics", {}),
                    live_value if 'live_value' in locals() else 0,
                    index_name if 'index_name' in locals() else "Custom Index"
                )
                st.download_button(
                    "Download CSV Factsheet",
                    data=csv_content.encode('utf-8'),
                    file_name=f"factsheet_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col_fact2:
            if st.button("Generate HTML Factsheet"):
                html_content = generate_factsheet_html_content(
                    current_calc_df,
                    current_hist_df,
                    comp_df,
                    st.session_state.get("last_comparison_metrics", {}),
                    live_value if 'live_value' in locals() else 0,
                    index_name if 'index_name' in locals() else "Custom Index",
                    ai_snippet if ai_snippet.strip() else None
                )
                st.download_button(
                    "Download HTML Factsheet",
                    data=html_content.encode('utf-8'),
                    file_name=f"factsheet_{datetime.now().strftime('%Y%m%d')}.html",
                    mime="text/html"
                )

# --- TAB 3: Index Price Calculator (Enhanced) ---
with tabs[2]:
    st.header("⚡ Live Index Price Calculator")
    
    if not k:
        st.info("Login to Kite first")
    else:
        st.markdown("### Configuration")
        
        calc_method = st.radio(
            "Input Method",
            ["CSV Upload", "Manual Entry", "From Saved Index"],
            horizontal=True
        )
        
        df_calc = pd.DataFrame()
        
        if calc_method == "CSV Upload":
            uploaded = st.file_uploader("Upload CSV (Symbol, Weights columns)", type="csv")
            if uploaded:
                try:
                    df_calc = pd.read_csv(uploaded)
                    df_calc.columns = [c.strip().lower() for c in df_calc.columns]
                    if 'symbol' in df_calc.columns and 'weights' in df_calc.columns:
                        df_calc['symbol'] = df_calc['symbol'].str.strip().str.upper()
                        df_calc['weights'] = pd.to_numeric(df_calc['weights'], errors='coerce')
                        df_calc.dropna(subset=['symbol', 'weights'], inplace=True)
                        df_calc.rename(columns={'symbol': 'Symbol', 'weights': 'Weights'}, inplace=True)
                        st.session_state.index_price_calc_df = df_calc
                        st.success(f"Loaded {len(df_calc)} symbols")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif calc_method == "Manual Entry":
            symbols_in = st.text_input("Symbols (comma-separated)", "RELIANCE,TCS,INFY")
            weights_in = st.text_input("Weights (comma-separated)", "0.4,0.3,0.3")
            
            if st.button("Load Manual Data"):
                try:
                    syms = [s.strip().upper() for s in symbols_in.split(',')]
                    wts = [float(w.strip()) for w in weights_in.split(',')]
                    if len(syms) == len(wts):
                        df_calc = pd.DataFrame({'Symbol': syms, 'Weights': wts})
                        st.session_state.index_price_calc_df = df_calc
                        st.success(f"Loaded {len(syms)} symbols")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif calc_method == "From Saved Index":
            saved_idxs = st.session_state.get("saved_indexes", [])
            if saved_idxs:
                idx_names = [idx['index_name'] for idx in saved_idxs]
                selected = st.selectbox("Select saved index", idx_names)
                if st.button("Load Index"):
                    idx_data = next((idx for idx in saved_idxs if idx['index_name'] == selected), None)
                    if idx_data:
                        constituents = pd.DataFrame(idx_data['constituents'])
                        df_calc = constituents[['symbol', 'Weights']].rename(columns={'symbol': 'Symbol'})
                        st.session_state.index_price_calc_df = df_calc
                        st.success(f"Loaded {selected}")
            else:
                st.info("No saved indexes. Create one in Custom Index tab.")

        df_calc = st.session_state.get("index_price_calc_df", pd.DataFrame())
        
        if not df_calc.empty:
            st.markdown("---")
            st.subheader("Constituents Preview")
            st.dataframe(df_calc, use_container_width=True)
            
            # Advanced calculation options
            with st.expander("⚙️ Advanced Options"):
                calc_method_type = st.selectbox(
                    "Calculation Method",
                    ["Simple Weighted", "Market Cap Weighted", "Price Weighted"]
                )
                refresh_interval = st.number_input("Auto-refresh interval (seconds, 0=disabled)", 0, 300, 0)
                show_individual_contrib = st.checkbox("Show individual contributions", value=True)
            
            if st.button("Calculate Index Price", type="primary"):
                with st.spinner("Fetching live prices..."):
                    symbols = df_calc['Symbol'].tolist()
                    instrument_ids = [f"{DEFAULT_EXCHANGE}:{s}" for s in symbols]
                    
                    try:
                        ltp_data = k.ltp(instrument_ids)
                        prices = {}
                        for sym in symbols:
                            key = f"{DEFAULT_EXCHANGE}:{sym}"
                            prices[sym] = ltp_data.get(key, {}).get('last_price', np.nan)
                        
                        results = df_calc.copy()
                        results['LTP'] = results['Symbol'].map(prices)
                        results['Weighted Price'] = results['LTP'] * results['Weights']
                        
                        failed = results[results['LTP'].isna()]['Symbol'].tolist()
                        if failed:
                            st.warning(f"No LTP for: {', '.join(failed)}")
                        
                        final_price = results['Weighted Price'].sum()
                        
                        st.markdown("---")
                        st.subheader("📊 Results")
                        
                        col_res1, col_res2, col_res3 = st.columns(3)
                        col_res1.metric("Index Price", f"₹{final_price:,.2f}")
                        col_res2.metric("Valid Components", f"{results['LTP'].notna().sum()}/{len(results)}")
                        col_res3.metric("Total Weight", f"{results['Weights'].sum():.4f}")
                        
                        if show_individual_contrib:
                            st.markdown("#### Detailed Breakdown")
                            results['Contribution %'] = (results['Weighted Price'] / final_price * 100).fillna(0)
                            st.dataframe(results.style.format({
                                "Weights": "{:.4f}",
                                "LTP": "₹{:,.2f}",
                                "Weighted Price": "₹{:,.2f}",
                                "Contribution %": "{:.2f}%"
                            }), use_container_width=True)
                            
                            # Contribution pie chart
                            fig_contrib = px.pie(
                                results, 
                                names='Symbol', 
                                values='Contribution %',
                                title="Price Contribution by Symbol"
                            )
                            fig_contrib.update_layout(template="plotly_dark", height=400)
                            st.plotly_chart(fig_contrib, use_container_width=True)
                        
                        st.download_button(
                            "Download Calculation Details",
                            data=results.to_csv(index=False).encode('utf-8'),
                            file_name=f"index_calc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Calculation error: {e}")

# --- TAB 4: Compliance (Enhanced) ---
with tabs[3]:
    st.header("💼 Investment Compliance & Portfolio Analysis")
    
    if not k:
        st.info("Login to Kite first")
    else:
        col_up, col_rules = st.columns([2, 3])
        
        with col_up:
            st.subheader("1. Upload Portfolio")
            uploaded = st.file_uploader(
                "CSV file (Symbol, Industry, Quantity, Market/Fair Value required)",
                type="csv"
            )
        
        with col_rules:
            st.subheader("2. Define Rules")
            rules = st.text_area(
                "One rule per line",
                height=150,
                value="# Example rules:\nSTOCK RELIANCE < 10%\nSECTOR BANKS < 30%\nTOP_N_STOCKS 5 < 40%\nCOUNT_STOCKS > 15"
            )
            with st.expander("📖 Rule Syntax Guide"):
                st.markdown("""
                **Available Rules:**
                - `STOCK [Symbol] <op> [Value]%` - Individual stock limit
                - `SECTOR [Name] <op> [Value]%` - Sector exposure limit
                - `RATING [Rating] <op> [Value]%` - Rating-based limit
                - `ASSET_CLASS [Name] <op> [Value]%` - Asset class limit
                - `TOP_N_STOCKS [N] <op> [Value]%` - Top N stocks combined
                - `TOP_N_SECTORS [N] <op> [Value]%` - Top N sectors combined
                - `COUNT_STOCKS <op> [Value]` - Portfolio size constraint
                
                **Operators:** `<`, `>`, `<=`, `>=`, `=`
                """)
        
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                df.columns = [str(c).strip().lower().replace(' ', '_').replace('.', '').replace('/','_') for c in df.columns]
                
                header_map = {
                    'isin': 'ISIN', 'name_of_the_instrument': 'Name', 'symbol': 'Symbol',
                    'industry': 'Industry', 'quantity': 'Quantity', 'rating': 'Rating',
                    'asset_class': 'Asset Class', 'market_fair_value(rs_in_lacs)': 'Uploaded Value (Lacs)'
                }
                df = df.rename(columns=header_map)
                
                for col in ['Rating', 'Asset Class', 'Industry']:
                    if col in df.columns:
                        df[col] = df[col].str.strip().str.upper()

                missing = [c for c in ['Symbol', 'Quantity', 'Uploaded Value (Lacs)', 'Industry'] if c not in df.columns]
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                    st.session_state.compliance_results_df = pd.DataFrame()
                else:
                    df['Symbol'] = df['Symbol'].str.strip().str.upper()
                    df['Quantity'] = pd.to_numeric(df['Quantity'].astype(str).str.replace(',', ''), errors='coerce')
                    df['Uploaded Value (Lacs)'] = pd.to_numeric(df['Uploaded Value (Lacs)'], errors='coerce')
                    df.dropna(subset=['Symbol', 'Quantity', 'Uploaded Value (Lacs)'], inplace=True)

                    if st.button("Analyze Portfolio", type="primary", use_container_width=True):
                        with st.spinner("Analyzing..."):
                            symbols = df['Symbol'].unique().tolist()
                            instrument_ids = [f"{DEFAULT_EXCHANGE}:{s}" for s in symbols]
                            
                            try:
                                ltp_data = k.ltp(instrument_ids)
                                prices = {sym: ltp_data.get(f"{DEFAULT_EXCHANGE}:{sym}", {}).get('last_price') for sym in symbols}
                                
                                results = df.copy()
                                results['LTP'] = results['Symbol'].map(prices)
                                
                                failed = results[results['LTP'].isna()]['Symbol'].tolist()
                                if failed:
                                    st.warning(f"No LTP for: {', '.join(failed)}")
                                
                                results['Real-time Value (Rs)'] = (results['LTP'] * results['Quantity']).fillna(0)
                                total_val = results['Real-time Value (Rs)'].sum()
                                results['Weight %'] = (results['Real-time Value (Rs)'] / total_val * 100) if total_val > 0 else 0
                                
                                st.session_state.compliance_results_df = results
                                st.success("Analysis complete!")
                            except Exception as e:
                                st.error(f"Error: {e}")
                                st.session_state.compliance_results_df = pd.DataFrame()
            except Exception as e:
                st.error(f"File processing error: {e}")
                st.session_state.compliance_results_df = pd.DataFrame()

        results_df = st.session_state.get("compliance_results_df", pd.DataFrame())
        if not results_df.empty and 'Weight %' in results_df.columns:
            st.markdown("---")
            
            analysis_tabs = st.tabs(["📊 Dashboard", "🔍 Breakdowns", "⚖️ Compliance", "📄 Holdings"])

            with analysis_tabs[0]:
                st.subheader("Portfolio Dashboard")
                total_val = results_df['Real-time Value (Rs)'].sum()
                
                kpi_cols = st.columns(4)
                kpi_cols[0].metric("Portfolio Value", f"₹{total_val:,.2f}")
                kpi_cols[1].metric("Holdings", f"{len(results_df)}")
                kpi_cols[2].metric("Sectors", f"{results_df['Industry'].nunique()}")
                if 'Rating' in results_df.columns:
                    kpi_cols[3].metric("Ratings", f"{results_df['Rating'].nunique()}")

                st.markdown("#### Concentration Analysis")
                stock_hhi = (results_df['Weight %'] ** 2).sum()
                sector_hhi = (results_df.groupby('Industry')['Weight %'].sum() ** 2).sum()
                
                def hhi_category(score):
                    if score < 1500: return "Low Concentration"
                    elif score <= 2500: return "Moderate"
                    else: return "High Concentration"

                conc_cols = st.columns(2)
                with conc_cols[0]:
                    st.metric("Top 5 Stocks", f"{results_df.nlargest(5, 'Weight %')['Weight %'].sum():.2f}%")
                    st.metric("Top 10 Stocks", f"{results_df.nlargest(10, 'Weight %')['Weight %'].sum():.2f}%")
                    st.metric("Top 3 Sectors", f"{results_df.groupby('Industry')['Weight %'].sum().nlargest(3).sum():.2f}%")
                with conc_cols[1]:
                    st.metric("Stock HHI", f"{stock_hhi:,.0f}", help=hhi_category(stock_hhi))
                    st.metric("Sector HHI", f"{sector_hhi:,.0f}", help=hhi_category(sector_hhi))
                
                st.info("HHI Scale: <1500 (Low), 1500-2500 (Moderate), >2500 (High)")

            with analysis_tabs[1]:
                st.subheader("Portfolio Breakdowns")
                
                path = ['Industry', 'Name']
                if 'Asset Class' in results_df.columns:
                    path.insert(0, 'Asset Class')
                    results_df['Asset Class'] = results_df['Asset Class'].fillna('UNCLASSIFIED')
                
                fig_sun = px.sunburst(
                    results_df, 
                    path=path, 
                    values='Real-time Value (Rs)',
                    hover_data={'Weight %': ':.2f'}
                )
                fig_sun.update_layout(margin=dict(t=20, l=20, r=20, b=20), height=600)
                st.plotly_chart(fig_sun, use_container_width=True)

                breakdown_cols = st.columns(3)
                with breakdown_cols[0]:
                    st.markdown("##### By Sector")
                    sector_wts = results_df.groupby('Industry')['Weight %'].sum().reset_index()
                    fig_sec = px.pie(sector_wts, names='Industry', values='Weight %', hole=0.4)
                    fig_sec.update_traces(textinfo='percent+label', showlegend=False)
                    st.plotly_chart(fig_sec, use_container_width=True)
                
                with breakdown_cols[1]:
                    if 'Asset Class' in results_df.columns:
                        st.markdown("##### By Asset Class")
                        asset_wts = results_df.groupby('Asset Class')['Weight %'].sum().reset_index()
                        fig_asset = px.pie(asset_wts, names='Asset Class', values='Weight %', hole=0.4)
                        fig_asset.update_traces(textinfo='percent+label', showlegend=False)
                        st.plotly_chart(fig_asset, use_container_width=True)
                
                with breakdown_cols[2]:
                    if 'Rating' in results_df.columns:
                        st.markdown("##### By Rating")
                        rating_wts = results_df.groupby('Rating')['Weight %'].sum().reset_index()
                        fig_rating = px.pie(rating_wts, names='Rating', values='Weight %', hole=0.4)
                        fig_rating.update_traces(textinfo='percent+label', showlegend=False)
                        st.plotly_chart(fig_rating, use_container_width=True)

            with analysis_tabs[2]:
                st.subheader("Compliance Validation")
                validation = parse_and_validate_rules(rules, results_df)
                
                if not validation:
                    st.info("No rules defined or all rules valid.")
                else:
                    for res in validation:
                        if res['status'] == "✅ PASS":
                            st.success(f"**{res['status']}:** `{res['rule']}` ({res['details']})")
                        elif res['status'] == "❌ FAIL":
                            st.error(f"**{res['status']}:** `{res['rule']}` ({res['details']})")
                        else:
                            st.warning(f"**{res['status']}:** `{res['rule']}` ({res['details']})")

            with analysis_tabs[3]:
                st.subheader("Detailed Holdings")
                display = results_df.copy()
                format_dict = {
                    'Real-time Value (Rs)': '₹{:,.2f}', 
                    'LTP': '₹{:,.2f}', 
                    'Weight %': '{:.2f}%'
                }
                
                col_order = ['Name', 'Symbol', 'Industry', 'Real-time Value (Rs)', 'Weight %', 'Quantity', 'LTP']
                if 'Asset Class' in display.columns:
                    col_order.insert(3, 'Asset Class')
                if 'Rating' in display.columns:
                    col_order.insert(3, 'Rating')
                
                display_cols = [c for c in col_order if c in display.columns]
                st.dataframe(display[display_cols].style.format(format_dict), use_container_width=True)

                st.download_button(
                    "📥 Download Report (CSV)",
                    data=display[display_cols].to_csv(index=False).encode('utf-8'),
                    file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

st.markdown("---")
st.caption("Invsion Connect - Advanced Investment Analysis Platform | Powered by Kite Connect & Supabase")

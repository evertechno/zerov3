import streamlit as st
from SmartApi.smartConnect import SmartConnect
import pyotp
import time
from typing import Optional, Dict, Any

# ------------- Utility / SmartAPI wrapper -------------

class SmartAPIClient:
    def __init__(self, api_key: str, client_code: str, password: str, totp_secret: str):
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_secret = totp_secret

        self.smart = SmartConnect(api_key=self.api_key)
        self.jwt_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.feed_token: Optional[str] = None
        self.logged_in = False

    def login(self) -> Dict[str, Any]:
        totp = pyotp.TOTP(self.totp_secret).now()
        resp = self.smart.generateSession(self.client_code, self.password, totp)
        if not resp.get("status", False):
            raise Exception(f"Login failed: {resp}")
        data = resp["data"]
        self.jwt_token = data.get("jwtToken")
        self.refresh_token = data.get("refreshToken")
        self.feed_token = self.smart.getfeedToken()
        self.logged_in = True
        return resp

    def refresh(self) -> Dict[str, Any]:
        if self.refresh_token is None:
            raise Exception("No refresh token available")
        resp = self.smart.generateToken(self.refresh_token)
        if not resp.get("status", False):
            raise Exception(f"Token refresh failed: {resp}")
        data = resp["data"]
        self.jwt_token = data.get("jwtToken")
        self.feed_token = data.get("feedToken")
        return resp

    def ensure_valid_token(self):
        pass

    def get_profile(self) -> Dict[str, Any]:
        return self.smart.getProfile(self.refresh_token)

    def get_quote(self, exchange: str, tradingsymbol: str, symboltoken: str) -> Dict[str, Any]:
        return self.smart.ltpData(exchange, tradingsymbol, symboltoken)

    def get_market_limits(self) -> Dict[str, Any]:
        return self.smart.rmsLimit()

    def get_order_book(self) -> Dict[str, Any]:
        return self.smart.orderBook()

    def get_positions(self) -> Dict[str, Any]:
        return self.smart.position()

    def get_holdings(self) -> Dict[str, Any]:
        return self.smart.holding()

    def place_order(self, orderparams: dict) -> Any:
        return self.smart.placeOrder(orderparams)

    def modify_order(self, params: dict) -> Any:
        return self.smart.modifyOrder(params)

    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> Any:
        return self.smart.cancelOrder(order_id, variety)

    def logout(self) -> Dict[str, Any]:
        return self.smart.terminateSession(self.client_code)


# ------------- Streamlit App --------------

def load_credentials_from_secrets():
    secrets = st.secrets.get("smartapi", None)
    if not secrets:
        st.error("SmartAPI credentials not found in secrets.toml under [smartapi]")
        return None
    required = ["api_key", "client_code", "password", "totp_secret"]
    for k in required:
        if k not in secrets:
            st.error(f"Missing key {k} in smartapi secrets")
            return None
    return secrets


def main():
    st.set_page_config(page_title="SmartAPI Dashboard", layout="wide")
    st.title("📈 SmartAPI / Angel Broking Dashboard")

    if "user_logged_in" not in st.session_state:
        st.session_state.user_logged_in = False
    if "api_client" not in st.session_state:
        st.session_state.api_client: Optional[SmartAPIClient] = None

    if not st.session_state.user_logged_in:
        st.subheader("Login")
        user_email = st.text_input("Email")
        user_pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if user_email == "test@example.com" and user_pwd == "password":
                st.session_state.user_logged_in = True
                st.success("User login successful")
                st.experimental_rerun()
            else:
                st.error("Wrong credentials")
        return

    st.sidebar.success("User logged in")
    if st.sidebar.button("Logout"):
        st.session_state.user_logged_in = False
        st.session_state.api_client = None
        st.experimental_rerun()

    if st.session_state.api_client is None:
        creds = load_credentials_from_secrets()
        if creds:
            cli = SmartAPIClient(
                api_key=creds["api_key"],
                client_code=creds["client_code"],
                password=creds["password"],
                totp_secret=creds["totp_secret"]
            )
            try:
                cli.login()
                st.session_state.api_client = cli
                st.success("SmartAPI login successful")
            except Exception as e:
                st.error(f"SmartAPI login error: {e}")
                return
        else:
            return

    client = st.session_state.api_client

    tab_profile, tab_quotes, tab_orders, tab_market = st.tabs(
        ["Profile", "Quotes", "Orders & Portfolio", "Market Data"]
    )

    with tab_profile:
        st.subheader("User Profile")
        try:
            profile = client.get_profile()
            st.json(profile)
        except Exception as e:
            st.error(f"Error fetching profile: {e}")

    with tab_quotes:
        st.subheader("Fetch Quote / LTP")
        col1, col2, col3 = st.columns(3)
        with col1:
            exchange = st.text_input("Exchange", value="NSE")
        with col2:
            tradingsymbol = st.text_input("Trading Symbol", value="INFY-EQ")
        with col3:
            symboltoken = st.text_input("Symbol Token", value="")
        if st.button("Get Quote"):
            try:
                quote = client.get_quote(exchange, tradingsymbol, symboltoken)
                st.json(quote)
            except Exception as e:
                st.error(f"Error fetching quote: {e}")

    with tab_orders:
        st.subheader("Order / Positions / Holdings")
        if st.button("Show Order Book"):
            try:
                st.json(client.get_order_book())
            except Exception as e:
                st.error(f"Error order book: {e}")

        if st.button("Show Positions"):
            try:
                st.json(client.get_positions())
            except Exception as e:
                st.error(f"Error positions: {e}")

        if st.button("Show Holdings"):
            try:
                st.json(client.get_holdings())
            except Exception as e:
                st.error(f"Error holdings: {e}")

        st.markdown("---")
        st.subheader("Place Order (LIMIT)")

        with st.form("order_form"):
            var = st.selectbox("Variety", ["NORMAL", "AMO"])
            exch = st.selectbox("Exchange", ["NSE", "BSE"])
            ts = st.text_input("Trading Symbol", "")
            stkn = st.text_input("Symbol Token", "")
            txn = st.selectbox("Transaction Type", ["BUY", "SELL"])
            ordert = st.selectbox("Order Type", ["LIMIT", "MARKET"])
            prod = st.selectbox("Product Type", ["INTRADAY", "DELIVERY", "MARGIN"])
            dur = st.selectbox("Duration", ["DAY", "IOC", "GTC"])
            price = st.text_input("Price (for LIMIT)", "")
            qty = st.text_input("Quantity", "")
            sl = st.text_input("Stop Loss", "0")
            so = st.text_input("Square off", "0")

            btn = st.form_submit_button("Place Order")
            if btn:
                params = {
                    "variety": var,
                    "exchange": exch,
                    "tradingsymbol": ts,
                    "symboltoken": stkn,
                    "transactiontype": txn,
                    "ordertype": ordert,
                    "producttype": prod,
                    "duration": dur,
                    "price": price,
                    "stoploss": sl,
                    "squareoff": so,
                    "quantity": qty
                }
                try:
                    res = client.place_order(params)
                    st.success(f"Order placed: {res}")
                except Exception as e:
                    st.error(f"Error placing order: {e}")

    with tab_market:
        st.subheader("Market / RMS Limits")
        if st.button("Fetch RMS Limits"):
            try:
                st.json(client.get_market_limits())
            except Exception as e:
                st.error(f"Error fetching limits: {e}")


if __name__ == "__main__":
    main()

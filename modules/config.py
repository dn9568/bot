# modules/config.py
import os
# -------------------------------------------------------------------
# Binance Mainnet API credentials (hardcoded for local use)
API_KEY           = "zv3gmlj0nzqXO2kbObfLgkXF3iAppLzbt4yuXESiHtwqgP4keRv17vAB3ETa2qIa"
API_SECRET        = "aKPngvvNtMmHKuv6E01v1XAyJ4td1tK47HyoGHUBarz3o3Pg1t3EC56nmByM4tWw"

BINANCE_API_KEY    = "zv3gmlj0nzqXO2kbObfLgkXF3iAppLzbt4yuXESiHtwqgP4keRv17vAB3ETa2qIa"
BINANCE_API_SECRET = "aKPngvvNtMmHKuv6E01v1XAyJ4td1tK47HyoGHUBarz3o3Pg1t3EC56nmByM4tWw"


ALPACA_PAPER_API_KEY    = "PK4YV12AODTH8XBLCA5R"
ALPACA_PAPER_API_SECRET = "UHPYQAu6YRqNtJ3oBrku8oh38BxbN9MWG4BPohVp"
ALPACA_PAPER_BASE_URL   = "https://paper-api.alpaca.markets"

# ใส่ Key ที่ได้จาก Dashboard ของ Alpaca Paper Trading ที่นี่
APCA_API_KEY = "PK4YV12AODTH8XBLCA5R"
APCA_API_SECRET = "UHPYQAu6YRqNtJ3oBrku8oh38BxbN9MWG4BPohVp"
APCA_BASE_URL = "https://paper-api.alpaca.markets" # ไม่ต้องเปลี่ยน

# Bot Operation Mode
# Always use real mainnet data; orders are simulated locally
TESTNET           = False

# --- File Paths ---
NEWS_CACHE_FILE   = "news_cache.json"
DATA_FOLDER       = "ai_models"
LOG_FOLDER        = "logs"
AI_MODEL_FOLDER   = "ai_models"

# --- Reinforcement Learning Model ---
# Name of the extracted model folder under ai_models/
RL_MODEL_NAME     = "ppo_rl_model_advanced"
# Full path to RL model checkpoint
RL_MODEL_PATH     = os.path.join(AI_MODEL_FOLDER, RL_MODEL_NAME)

# --- Asset Universe & Timeframes ---
TARGET_PAIRS       = ["BTCUSDT","ETHUSDT","LTCUSDT"]

#TARGET_PAIRS      = [
#    {"symbol": "BTC/USDT", "category": "future"},
#    {"symbol": "ETH/USDT", "category": "future"},
#    {"symbol": "BNB/USDT", "category": "future"},
#    {"symbol": "LTC/USDT", "category": "future"},
#]
TIMEFRAME         = '5m'
TREND_TIMEFRAME   = '1h'
INITIAL_FETCH_LIMIT = 1000

# --- Trading & Risk Management ---
INITIAL_BALANCE_USDT = 10_000.0      # เริ่มต้นยอดกระดาษ
TRADE_AMOUNT_USDT = 100.0
MAX_TRADE_AMOUNT_USDT = 100    # maximum USDT margin per position
MIN_TRADE_AMOUNT_USDT = 20     # minimum USDT margin per position
LEVERAGE              = 20
MAX_OPEN_POSITIONS    = 1
RISK_PERCENT          = 0.01   # fraction of balance per trade
START_BALANCE      = 10000.0


# --- Order Management ---
ORDER_CANCEL_SECONDS   = 600    # auto-cancel unfilled orders after 10m
AGGRESSION_PCT         = 0.0005 # fraction to adjust limit orders

# --- Dynamic Interval Settings ---
DYNAMIC_INTERVAL_ENABLED = True
VOLATILITY_ATR_THRESHOLD  = 0.02  # 2% ATR/Price ratio
HIGH_VOL_INTERVAL_SEC     = 300   # 15 minutes
LOW_VOL_INTERVAL_SEC      = 3600  # 1 hour

# --- Take-Profit & Stop-Loss ---
USE_DYNAMIC_TP          = True
INITIAL_TP_THRESHOLD    = 0.15   # activate trailing stop at 15% profit
TRAILING_STOP_PCT       = 0.05   # 5% drop from peak profit
TAKE_PROFIT_THRESHOLD   = 0.20   # hard TP at 20% absolute profit
STOP_LOSS_THRESHOLD     = -0.20  # hard SL at 20% loss

# --- Indicator & AI Model Params ---
SMA_PERIODS    = [50, 200]
EMA_PERIODS    = [12, 26]
RSI_PERIOD     = 14
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9
BBANDS_PERIOD  = 20
BBANDS_STD_DEV = 2
ATR_PERIOD     = 14
CCI_PERIOD     = 20
PSAR_ACCELERATION    = 0.02
PSAR_MAX_ACCELERATION= 0.2
STD_DEV_PERIOD       = 20
OBV_SMA_PERIOD       = 20

LSTM_SEQ_LEN       = 50
LSTM_UNITS         = 32
LSTM_EPOCHS        = 5
LSTM_BATCH_SIZE    = 32
AE_UNITS           = 32
AE_LATENT_DIM      = 16
AE_EPOCHS          = 10
AE_BATCH_SIZE      = 32
ANOMALY_THRESHOLD  = 0.01

RSI_OVERSOLD       = 30
RSI_OVERBOUGHT     = 70

MODEL_FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
] + [f"SMA_{p}" for p in SMA_PERIODS] + [f"EMA_{p}" for p in EMA_PERIODS] + \
[f"RSI_{RSI_PERIOD}"] + ['MACD_line', 'MACD_signal', 'MACD_hist'] + \
['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'] + [f'ATR_{ATR_PERIOD}'] + \
[f'CCI_{CCI_PERIOD}_0.015'] + ['PSAR_long', 'PSAR_short'] + \
[f'STDEV_{STD_DEV_PERIOD}'] + ['OBV', f'OBV_SMA_{OBV_SMA_PERIOD}']

MODEL_WEIGHTS = {
    'lstm':      1.0,
    'sentiment': 0.8,
    'anomaly':   1.5,
    'rsi':       0.5,
    'macd':      0.7,
    'trend':     1.2,
    'bbands':    0.6,
    'cci':       0.5,
    'psar':      0.8,
    'obv':       0.4,
}
DECISION_THRESHOLD_BUY  = 0.5
DECISION_THRESHOLD_SELL = 0.5

# Execution Scheduling
SLEEP_INTERVAL    = 60
UPDATE_INTERVAL_SEC = SLEEP_INTERVAL


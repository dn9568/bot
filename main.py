# main.py

import os
import sys
import time
import csv
import pickle
import json
import logging
from datetime import datetime

# ให้ Python หาโมดูลจากโฟลเดอร์ modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from config import (
    TARGET_PAIRS,
    INITIAL_BALANCE_USDT,
    TRADE_AMOUNT_USDT,
    SLEEP_INTERVAL,
    LOG_FOLDER,
    DATA_FOLDER,
    BINANCE_API_KEY,
    BINANCE_API_SECRET
)
from binance_client import create_binance_data_client, get_latest_prices
from executor import BinancePaperExecutor
from ai_decider import AdvancedAIPredictor

# ── เตรียมโฟลเดอร์ logs & data ─────────────────────────────────────────
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
csv_path = os.path.join(LOG_FOLDER, "bot_log.csv")
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "symbol", "action", "trade_price", "trade_amount_usdt",
            "balance", "position", "pnl", "close_price"
        ])

# ตั้งค่า logging
logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "bot.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ── Warm-up AI Predictor ─────────────────────────────────────────────────
predictor = AdvancedAIPredictor()
history_file = os.path.join(DATA_FOLDER, "price_history.pkl")
if os.path.exists(history_file):
    try:
        with open(history_file, "rb") as f:
            hist = pickle.load(f)
        for symbol, prices in hist.items():
            for p in prices:
                predictor.update_price_history(symbol, p)
        print(f"✅ Loaded historical data for AI predictor ({len(hist)} symbols)")
    except Exception as e:
        print(f"❌ Error loading history.pkl: {e}")

# ── สร้าง Binance client + Paper Executor ────────────────────────────────
data_client = create_binance_data_client(
    api_key=BINANCE_API_KEY,
    api_secret=BINANCE_API_SECRET
)
executor = BinancePaperExecutor(
    initial_balance=INITIAL_BALANCE_USDT,
    trade_amount=TRADE_AMOUNT_USDT
)

prev_nav = None

# ── Main Loop ──────────────────────────────────────────────────────────────
while True:
    try:
        # 1) Fetch real prices from Binance Mainnet
        prices = get_latest_prices(data_client, TARGET_PAIRS)
        print("Fetching crypto prices from Binance...", flush=True)
        for sym, price in prices.items():
            print(f"✅ {sym}: ${price:.2f}", flush=True)
            predictor.update_price_history(sym, price)

        # Save updated history
        try:
            with open(history_file, "wb") as f:
                pickle.dump(dict(predictor.history), f)
            print(f"✅ Saved price history to {history_file}", flush=True)
        except Exception as e:
            logging.error(f"❌ Could not save history: {e}")

        # 2) AI Decision
        symbol, action = predictor.select_and_decide(prices)
        trade_price = prices.get(symbol, 0.0)
        print(f"AI Decision: {symbol}, Action: {action}, Price: {trade_price:.2f}", flush=True)

        # 3) Execute paper trade
        result = executor.execute(action, symbol, trade_price)

        # 4) Compute NAV & PnL
        bal = result.get("balance", 0.0) or 0.0
        pos = result.get("position", 0.0) or 0.0
        nav = bal + pos * trade_price
        pnl = nav - prev_nav if prev_nav is not None else 0.0
        prev_nav = nav
        close_price = trade_price if action == -1 else ""

        # 5) Log to CSV
        timestamp = datetime.utcnow().isoformat()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, symbol, action,
                f"{trade_price:.2f}", TRADE_AMOUNT_USDT,
                bal, pos, f"{pnl:.2f}", close_price
            ])

        # 6) Log JSON record to JSONL, file, and console
        record = {
            "timestamp": timestamp,
            "symbol": symbol,
            "action": action,
            "trade_price": round(trade_price, 2),
            "trade_amount_usdt": TRADE_AMOUNT_USDT,
            "balance": round(bal, 2),
            "position": pos,
            "pnl": round(pnl, 2),
            "close_price": close_price
        }
        json_log_path = os.path.join(LOG_FOLDER, "bot_log.jsonl")
        try:
            with open(json_log_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logging.error(f"❌ Could not write JSON log: {e}")

        logging.info(json.dumps(record, ensure_ascii=False))
        print(json.dumps(record, ensure_ascii=False), flush=True)

    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user", flush=True)
        break
    except Exception as e:
        logging.error(f"❌ Error in main loop: {e}", exc_info=True)

    # Sleep until next iteration
    print(f"⏳ Sleeping {SLEEP_INTERVAL}s…", flush=True)
    time.sleep(SLEEP_INTERVAL)

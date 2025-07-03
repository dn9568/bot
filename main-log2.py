# futures_trading_bot_enhanced.py - With Complete Logging System
# =============================================================
import os
import sys
import time
import csv
import pickle
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
import asyncio
from typing import Dict, List, Tuple, Optional
import traceback
import joblib
from train_enhanced_ai import SimpleAITrainer

# Add modules path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

warnings.filterwarnings('ignore')

# Import configs
from config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    LOG_FOLDER,
    DATA_FOLDER
)

# กำหนดพาธโมเดลใหม่
MODEL_DIR = 'ai_models/simple_enhanced'
SCALER_PATH = f'{MODEL_DIR}/scaler.pkl'
FEATURES_PATH = f'{MODEL_DIR}/features.json'
MODEL_FILES = [f for f in os.listdir(MODEL_DIR) if f.endswith('_model.pkl')]

# โหลด scaler + รายชื่อ features
scaler = joblib.load(SCALER_PATH)
with open(FEATURES_PATH) as f:
    feature_names = json.load(f)

# โหลดโมเดลทั้งหมดเป็น ensemble
models = [joblib.load(f'{MODEL_DIR}/{m}') for m in MODEL_FILES]


# Import Binance client
from binance.client import Client

# ===================== Enhanced Configuration =====================

class FuturesConfig:
    """Enhanced Configuration for Futures Trading"""
    # Trading parameters
    INITIAL_BALANCE_USDT = 10000.0
    POSITION_SIZE_PCT = 2.0  # 2% per position
    MAX_POSITIONS = 3
    LEVERAGE = 10  # 10x leverage
    
    # Risk management
    TAKE_PROFIT_PCT = 2.0    # 2% TP
    STOP_LOSS_PCT = 1.0      # 1% SL
    MAX_DRAWDOWN_PCT = 15.0  # 15% max drawdown
    TRAILING_STOP_PCT = 1.5  # Trailing stop
    
    # Trading pairs
    SYMBOLS = [
        'BTCUSDT',
        'ETHUSDT',
        'BNBUSDT',
        'SOLUSDT',
        'LTCUSDT',
        'BCHUSDT'
    ]
    
    # API settings
    UPDATE_INTERVAL = 30  # seconds
    
    # News analysis
    USE_NEWS_ANALYSIS = True
    NEWS_IMPACT_WEIGHT = 0.3
    
    # Logging settings
    LOG_TRADES = True
    LOG_ANALYSIS = True
    LOG_PORTFOLIO = True
    SAVE_INTERVAL = 300  # Save logs every 5 minutes

# ===================== Comprehensive Logger =====================

class TradingLogger:
    """Comprehensive logging system for trading bot"""
    
    def __init__(self, log_folder: str):
        self.log_folder = log_folder
        os.makedirs(log_folder, exist_ok=True)
        
        # Create dated subfolders
        self.date_folder = os.path.join(log_folder, datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(self.date_folder, exist_ok=True)
        
        # Initialize log files
        self.init_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Different log files
        self.trade_log_file = os.path.join(self.date_folder, f'trades_{self.init_time}.csv')
        self.analysis_log_file = os.path.join(self.date_folder, f'analysis_{self.init_time}.csv')
        self.portfolio_log_file = os.path.join(self.date_folder, f'portfolio_{self.init_time}.csv')
        self.summary_log_file = os.path.join(self.date_folder, f'summary_{self.init_time}.json')
        self.error_log_file = os.path.join(self.date_folder, f'errors_{self.init_time}.txt')
        
        # Real-time logs
        self.realtime_log = os.path.join(self.date_folder, f'realtime_{self.init_time}.txt')
        
        # Initialize CSV headers
        self._init_csv_files()
        
        # Setup Python logging
        self._setup_python_logging()
        
        # Performance metrics
        self.metrics = {
            'start_time': datetime.now(),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'max_profit': 0,
            'max_loss': 0,
            'total_fees': 0,
            'errors': []
        }
    
    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Trade log header
        with open(self.trade_log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'action', 'symbol', 'side', 'price', 'size',
                'margin_used', 'pnl', 'pnl_pct', 'balance_after', 'reason',
                'confidence', 'hold_time', 'exit_reason'
            ])
        
        # Analysis log header
        with open(self.analysis_log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'symbol', 'price', 'rsi', 'macd', 'bb_position',
                'momentum', 'volatility', 'trend', 'sentiment', 'recommendation',
                'confidence', 'news_impact'
            ])
        
        # Portfolio log header
        with open(self.portfolio_log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'free_balance', 'used_margin', 'unrealized_pnl',
                'total_nav', 'drawdown_pct', 'open_positions', 'win_rate',
                'sharpe_ratio', 'total_return_pct'
            ])
    
    def _setup_python_logging(self):
        """Setup Python's logging module"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.error_log_file),
                logging.StreamHandler()
            ]
        )
    
    def log_trade(self, trade_data: Dict):
        """Log trade execution"""
        self.metrics['total_trades'] += 1
        
        if trade_data.get('pnl', 0) > 0:
            self.metrics['winning_trades'] += 1
            self.metrics['max_profit'] = max(self.metrics['max_profit'], trade_data['pnl'])
        else:
            self.metrics['losing_trades'] += 1
            self.metrics['max_loss'] = min(self.metrics['max_loss'], trade_data['pnl'])
        
        self.metrics['total_pnl'] += trade_data.get('pnl', 0)
        
        # Write to CSV
        with open(self.trade_log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade_data.get('timestamp', datetime.now()),
                trade_data.get('action', ''),
                trade_data.get('symbol', ''),
                trade_data.get('side', ''),
                trade_data.get('price', 0),
                trade_data.get('size', 0),
                trade_data.get('margin_used', 0),
                trade_data.get('pnl', 0),
                trade_data.get('pnl_pct', 0),
                trade_data.get('balance_after', 0),
                trade_data.get('reason', ''),
                trade_data.get('confidence', 0),
                trade_data.get('hold_time', 0),
                trade_data.get('exit_reason', '')
            ])
        
        # Log to realtime file
        self.log_realtime(f"TRADE: {trade_data}")
    
    def log_analysis(self, analysis_data: Dict):
        """Log market analysis"""
        with open(self.analysis_log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                analysis_data.get('timestamp', datetime.now()),
                analysis_data.get('symbol', ''),
                analysis_data.get('price', 0),
                analysis_data.get('rsi', 0),
                analysis_data.get('macd', 0),
                analysis_data.get('bb_position', 0),
                analysis_data.get('momentum', 0),
                analysis_data.get('volatility', 0),
                analysis_data.get('trend', ''),
                analysis_data.get('sentiment', 0),
                analysis_data.get('recommendation', ''),
                analysis_data.get('confidence', 0),
                analysis_data.get('news_impact', 0)
            ])
    
    def log_portfolio(self, portfolio_data: Dict):
        """Log portfolio status"""
        with open(self.portfolio_log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                portfolio_data.get('timestamp', datetime.now()),
                portfolio_data.get('free_balance', 0),
                portfolio_data.get('used_margin', 0),
                portfolio_data.get('unrealized_pnl', 0),
                portfolio_data.get('total_nav', 0),
                portfolio_data.get('drawdown_pct', 0),
                portfolio_data.get('open_positions', 0),
                portfolio_data.get('win_rate', 0),
                portfolio_data.get('sharpe_ratio', 0),
                portfolio_data.get('total_return_pct', 0)
            ])
    
    def log_error(self, error_msg: str):
        """Log errors"""
        timestamp = datetime.now()
        self.metrics['errors'].append({
            'timestamp': timestamp,
            'error': error_msg
        })
        
        logging.error(error_msg)
        self.log_realtime(f"ERROR: {error_msg}")
    
    def log_realtime(self, message: str):
        """Log real-time messages"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.realtime_log, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def save_summary(self):
        """Save summary statistics"""
        runtime = (datetime.now() - self.metrics['start_time']).total_seconds() / 3600
        
        summary = {
            'run_date': self.metrics['start_time'].strftime('%Y-%m-%d'),
            'runtime_hours': round(runtime, 2),
            'total_trades': self.metrics['total_trades'],
            'winning_trades': self.metrics['winning_trades'],
            'losing_trades': self.metrics['losing_trades'],
            'win_rate': round(self.metrics['winning_trades'] / max(self.metrics['total_trades'], 1) * 100, 2),
            'total_pnl': round(self.metrics['total_pnl'], 2),
            'max_profit': round(self.metrics['max_profit'], 2),
            'max_loss': round(self.metrics['max_loss'], 2),
            'avg_trade_pnl': round(self.metrics['total_pnl'] / max(self.metrics['total_trades'], 1), 2),
            'total_fees': round(self.metrics['total_fees'], 2),
            'errors_count': len(self.metrics['errors']),
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.summary_log_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    
    def create_daily_report(self):
        """Create daily performance report"""
        report_file = os.path.join(self.date_folder, f'daily_report_{self.init_time}.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("DAILY TRADING REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"Runtime: {(datetime.now() - self.metrics['start_time']).total_seconds() / 3600:.1f} hours\n\n")
            
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-"*30 + "\n")
            f.write(f"Total Trades: {self.metrics['total_trades']}\n")
            f.write(f"Win Rate: {self.metrics['winning_trades'] / max(self.metrics['total_trades'], 1) * 100:.1f}%\n")
            f.write(f"Total P&L: ${self.metrics['total_pnl']:.2f}\n")
            f.write(f"Best Trade: ${self.metrics['max_profit']:.2f}\n")
            f.write(f"Worst Trade: ${self.metrics['max_loss']:.2f}\n")
            
            if self.metrics['errors']:
                f.write(f"\nERRORS ({len(self.metrics['errors'])})\n")
                f.write("-"*30 + "\n")
                for err in self.metrics['errors'][-5:]:  # Last 5 errors
                    f.write(f"{err['timestamp'].strftime('%H:%M:%S')} - {err['error']}\n")

# ===================== News Analyzer =====================

class NewsAnalyzer:
    """Enhanced news analyzer with caching"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    async def get_news_sentiment(self, symbol: str) -> Dict:
        """Get news sentiment with caching"""
        # Check cache
        if symbol in self.cache:
            cached_time, cached_data = self.cache[symbol]
            if (datetime.now() - cached_time).seconds < self.cache_duration:
                return cached_data
        
        # Simulate news sentiment (replace with real API)
        sentiment_data = {
            'sentiment': np.random.uniform(-1, 1),
            'confidence': np.random.uniform(0.3, 0.8),
            'news_count': np.random.randint(0, 10),
            'summary': 'Demo news sentiment'
        }
        
        # Cache result
        self.cache[symbol] = (datetime.now(), sentiment_data)
        
        return sentiment_data

# ===================== Enhanced Position Manager =====================

class FuturesPositionManager:
    """Enhanced position manager with trailing stops"""
    
    def __init__(self, initial_balance: float, leverage: int, logger: TradingLogger):
        self.balance = initial_balance
        self.leverage = leverage
        self.positions = {}
        self.position_history = []
        self.total_pnl = 0
        self.liquidation_price_buffer = 0.8
        self.logger = logger
        
        # Track highest profit for trailing stop
        self.highest_profit = {}
    
    def calculate_position_size(self, price: float, balance_pct: float) -> float:
        """Calculate position size with leverage"""
        position_value = self.balance * (balance_pct / 100)
        position_size = (position_value * self.leverage) / price
        return position_size
    
    def calculate_liquidation_price(self, entry_price: float, side: str) -> float:
        """Calculate liquidation price"""
        margin_ratio = 1 / self.leverage
        if side == 'LONG':
            return entry_price * (1 - margin_ratio * self.liquidation_price_buffer)
        else:
            return entry_price * (1 + margin_ratio * self.liquidation_price_buffer)
    
    def open_position(self, symbol: str, side: str, entry_price: float, size: float, reason: str, confidence: float):
        """Open a new futures position with logging"""
        pos = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'size': size,
            'entry_time': datetime.utcnow(),
            'liquidation_price': self.calculate_liquidation_price(entry_price, side),
            'take_profit': self._calculate_tp(entry_price, side),
            'stop_loss': self._calculate_sl(entry_price, side),
            'unrealized_pnl': 0,
            'reason': reason,
            'confidence': confidence
        }
        
        self.positions[symbol] = pos
        margin_used = (size * entry_price) / self.leverage
        self.balance -= margin_used
        
        # Initialize trailing stop tracking
        self.highest_profit[symbol] = 0
        
        # Log trade
        self.logger.log_trade({
            'timestamp': datetime.now(),
            'action': 'OPEN',
            'symbol': symbol,
            'side': side,
            'price': entry_price,
            'size': size,
            'margin_used': margin_used,
            'pnl': 0,
            'pnl_pct': 0,
            'balance_after': self.balance,
            'reason': reason,
            'confidence': confidence,
            'hold_time': 0,
            'exit_reason': ''
        })
        
        emoji = '🟢' if side == 'LONG' else '🔴'
        print(f"\n{emoji} Opened {side} {symbol}: {size:.4f} @ ${entry_price:,.2f}")
        print(f"   TP ${pos['take_profit']:,.2f} | SL ${pos['stop_loss']:,.2f} | Liq ${pos['liquidation_price']:,.2f}")
        print(f"   Reason: {reason} (Confidence: {confidence:.1%})")
    
    def _calculate_tp(self, entry_price: float, side: str) -> float:
        if side == 'LONG':
            return entry_price * (1 + FuturesConfig.TAKE_PROFIT_PCT / 100)
        else:
            return entry_price * (1 - FuturesConfig.TAKE_PROFIT_PCT / 100)
    
    def _calculate_sl(self, entry_price: float, side: str) -> float:
        if side == 'LONG':
            return entry_price * (1 - FuturesConfig.STOP_LOSS_PCT / 100)
        else:
            return entry_price * (1 + FuturesConfig.STOP_LOSS_PCT / 100)
    
    def update_position_pnl(self, symbol: str, current_price: float) -> float:
        """Update P&L with trailing stop check"""
        if symbol not in self.positions:
            return 0
        
        pos = self.positions[symbol]
        
        if pos['side'] == 'LONG':
            pnl = (current_price - pos['entry_price']) * pos['size']
        else:
            pnl = (pos['entry_price'] - current_price) * pos['size']
        
        pos['unrealized_pnl'] = pnl
        
        # Update trailing stop
        if pnl > self.highest_profit[symbol]:
            self.highest_profit[symbol] = pnl
            
            # Adjust stop loss if profit is significant
            profit_pct = (pnl / ((pos['size'] * pos['entry_price']) / self.leverage)) * 100
            if profit_pct > FuturesConfig.TRAILING_STOP_PCT:
                # Move stop loss to breakeven or trailing level
                if pos['side'] == 'LONG':
                    new_sl = current_price * (1 - FuturesConfig.TRAILING_STOP_PCT / 100)
                    pos['stop_loss'] = max(pos['stop_loss'], new_sl, pos['entry_price'])
                else:
                    new_sl = current_price * (1 + FuturesConfig.TRAILING_STOP_PCT / 100)
                    pos['stop_loss'] = min(pos['stop_loss'], new_sl, pos['entry_price'])
        
        return pnl
    
    def should_close_position(self, symbol: str, current_price: float) -> Tuple[bool, str]:
        """Enhanced position close checks"""
        if symbol not in self.positions:
            return False, ""
        
        pos = self.positions[symbol]
        
        # Check liquidation
        if pos['side'] == 'LONG' and current_price <= pos['liquidation_price']:
            return True, f"LIQUIDATION @ ${current_price:,.2f}"
        if pos['side'] == 'SHORT' and current_price >= pos['liquidation_price']:
            return True, f"LIQUIDATION @ ${current_price:,.2f}"
        
        # Check take profit
        if pos['side'] == 'LONG' and current_price >= pos['take_profit']:
            return True, f"Take Profit @ ${current_price:,.2f}"
        if pos['side'] == 'SHORT' and current_price <= pos['take_profit']:
            return True, f"Take Profit @ ${current_price:,.2f}"
        
        # Check stop loss (including trailing)
        if pos['side'] == 'LONG' and current_price <= pos['stop_loss']:
            return True, f"Stop Loss @ ${current_price:,.2f}"
        if pos['side'] == 'SHORT' and current_price >= pos['stop_loss']:
            return True, f"Stop Loss @ ${current_price:,.2f}"
        
        # Time-based exit
        hold_hours = (datetime.utcnow() - pos['entry_time']).total_seconds() / 3600
        if hold_hours > 4:
            return True, f"Max hold time ({hold_hours:.1f}h)"
        
        return False, ""
    
    def close_position(self, symbol: str, exit_price: float, exit_reason: str) -> Dict:
        """Close position with comprehensive logging"""
        if symbol not in self.positions:
            return {}
        
        pos = self.positions.pop(symbol)
        
        # Calculate P&L
        if pos['side'] == 'LONG':
            pnl = (exit_price - pos['entry_price']) * pos['size']
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['size']
        
        margin = (pos['size'] * pos['entry_price']) / self.leverage
        self.balance += margin + pnl
        self.total_pnl += pnl
        pnl_pct = (pnl / margin) * 100
        
        # Calculate hold time
        hold_time = (datetime.utcnow() - pos['entry_time']).total_seconds() / 3600
        
        # Create trade record
        record = {
            'symbol': symbol,
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'size': pos['size'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_time': pos['entry_time'],
            'exit_time': datetime.utcnow(),
            'duration_hours': hold_time,
            'exit_reason': exit_reason
        }
        
        self.position_history.append(record)
        
        # Log trade
        self.logger.log_trade({
            'timestamp': datetime.now(),
            'action': 'CLOSE',
            'symbol': symbol,
            'side': pos['side'],
            'price': exit_price,
            'size': pos['size'],
            'margin_used': 0,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'balance_after': self.balance,
            'reason': pos['reason'],
            'confidence': pos['confidence'],
            'hold_time': hold_time,
            'exit_reason': exit_reason
        })
        
        # Remove from tracking
        del self.highest_profit[symbol]
        
        emoji = "✅" if pnl > 0 else "❌"
        print(f"\n{emoji} Closed {pos['side']} {symbol} @ ${exit_price:,.2f}")
        print(f"   P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%) | Hold: {hold_time:.1f}h")
        print(f"   Exit: {exit_reason}")
        
        return record

# ===================== Enhanced AI Predictor =====================

class FuturesAIPredictor:
    """Enhanced AI predictor with comprehensive analysis logging"""
    
    def __init__(self, logger: TradingLogger):
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        self.feature_cache = defaultdict(dict)
        self.min_history = 100
        self.news_analyzer = NewsAnalyzer()
        self.logger = logger
        self._load_models()
    
    def predict_signal(self, features: Dict) -> int:
        X   = np.array([features[n] for n in feature_names]).reshape(1, -1)
        Xs  = scaler.transform(X)
        probas = [m.predict_proba(Xs)[0] for m in models if hasattr(m, 'predict_proba')]
        if not probas:
            return 0
        avg   = np.mean(probas, axis=0)
        label = int(np.argmax(avg))
        return label - 1
        
    def _load_models(self):
        """Load ML models if available"""
        self.use_ml_models = False
        print("📊 Using technical analysis + news sentiment")
    
    def update_price(self, symbol: str, price: float):
        """Update price history"""
        self.price_history[symbol].append(price)
    
    def calculate_features(self, symbol: str) -> Optional[Dict]:
        """Calculate comprehensive technical features"""
        prices = list(self.price_history[symbol])
        if len(prices) < self.min_history:
            return None
        
        arr = np.array(prices)
        f = {}
        
        # Moving averages
        f['sma_20'] = np.mean(prices[-20:])
        f['sma_50'] = np.mean(prices[-50:])
        f['ema_9'] = self._calculate_ema(arr, 9)
        f['ema_21'] = self._calculate_ema(arr, 21)
        
        # Price ratios
        current_price = prices[-1]
        f['price_to_sma20'] = current_price / f['sma_20'] if f['sma_20'] > 0 else 1
        f['price_to_sma50'] = current_price / f['sma_50'] if f['sma_50'] > 0 else 1
        
        # Momentum indicators
        f['rsi_14'] = self._calculate_rsi(arr, 14)
        f['momentum_10'] = ((prices[-1] - prices[-11]) / prices[-11] * 100) if len(prices) > 10 else 0
        
        # Volatility
        if len(prices) >= 31:
            returns = np.diff(arr[-31:]) / arr[-31:-1]
            f['volatility'] = np.std(returns) * np.sqrt(365 * 48)  # Annualized
        else:
            f['volatility'] = 0.02
        
        # MACD
        ema12 = self._calculate_ema(arr, 12)
        ema26 = self._calculate_ema(arr, 26)
        f['macd'] = ema12 - ema26
        f['macd_signal'] = self._calculate_ema(np.array([f['macd']]), 9)
        f['macd_hist'] = f['macd'] - f['macd_signal']
        
        # Bollinger Bands
        bb_sma = f['sma_20']
        bb_std = np.std(prices[-20:])
        f['bb_upper'] = bb_sma + 2 * bb_std
        f['bb_lower'] = bb_sma - 2 * bb_std
        bb_range = f['bb_upper'] - f['bb_lower']
        f['bb_position'] = (current_price - f['bb_lower']) / bb_range if bb_range > 0 else 0.5
        
        # Support/Resistance
        f['resistance'] = np.max(prices[-50:])
        f['support'] = np.min(prices[-50:])
        
        # Trend detection
        if current_price > f['sma_50'] and f['sma_20'] > f['sma_50']:
            f['trend'] = 'UPTREND'
        elif current_price < f['sma_50'] and f['sma_20'] < f['sma_50']:
            f['trend'] = 'DOWNTREND'
        else:
            f['trend'] = 'SIDEWAYS'
        
        return f
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    async def predict_with_news(self, symbol: str, features: Dict) -> Tuple[str, float, str]:
        """Make prediction with comprehensive logging"""
        if not features:
            return 'NEUTRAL', 0.0, 'Insufficient data'
        
        # Technical analysis scoring
        ta_score = 0
        ta_conf = 0
        
        # Trend analysis
        if features['trend'] == 'UPTREND':
            ta_score += 1.5
            ta_conf += 0.4
        elif features['trend'] == 'DOWNTREND':
            ta_score -= 1.5
            ta_conf += 0.4
        
        # RSI analysis
        if features['rsi_14'] < 30:
            ta_score += 1.5
            ta_conf += 0.25
        elif features['rsi_14'] > 70:
            ta_score -= 1.5
            ta_conf += 0.25
        
        # MACD analysis
        if features['macd_hist'] > 0 and features['macd'] > features['macd_signal']:
            ta_score += 0.5
            ta_conf += 0.15
        elif features['macd_hist'] < 0 and features['macd'] < features['macd_signal']:
            ta_score -= 0.5
            ta_conf += 0.15
        
        # Bollinger Bands
        if features['bb_position'] < 0.2:
            ta_score += 0.5
            ta_conf += 0.1
        elif features['bb_position'] > 0.8:
            ta_score -= 0.5
            ta_conf += 0.1
        
        # Momentum
        if features['momentum_10'] > 5:
            ta_score += 0.3
            ta_conf += 0.1
        elif features['momentum_10'] < -5:
            ta_score -= 0.3
            ta_conf += 0.1
        
        # Get news sentiment
        news = await self.news_analyzer.get_news_sentiment(symbol)
        ns = news['sentiment']
        nc = news['confidence']
        
        # Combine TA and news
        if FuturesConfig.USE_NEWS_ANALYSIS and nc > 0.3:
            combined_score = ta_score * (1 - FuturesConfig.NEWS_IMPACT_WEIGHT) + ns * 3 * FuturesConfig.NEWS_IMPACT_WEIGHT
            combined_conf = (ta_conf + nc) / 2
        else:
            combined_score = ta_score
            combined_conf = ta_conf
        
        # Determine action
        if combined_score > 0.8 and combined_conf > 0.3:
            action = 'LONG'
        if features['trend'] == 'DOWNTREND' and combined_conf > 0.6:
            action = 'SHORT'
        elif combined_score < -0.8 and combined_conf > 0.3:
            action = 'SHORT'
        else:
            action = 'NEUTRAL'
        
        confidence = min(combined_conf, 0.9)
        
        # Generate reasoning
        reasons = []
        if features['rsi_14'] < 30:
            reasons.append(f"RSI oversold ({features['rsi_14']:.1f})")
        elif features['rsi_14'] > 70:
            reasons.append(f"RSI overbought ({features['rsi_14']:.1f})")
        
        if abs(features['momentum_10']) > 5:
            reasons.append(f"Momentum {features['momentum_10']:+.1f}%")
        
        if features['trend'] != 'SIDEWAYS':
            reasons.append(f"{features['trend']}")
        
        if nc > 0.3:
            reasons.append(f"News: {ns:+.2f}")
        
        reasoning = " | ".join(reasons[:3])
        
        # Log analysis
        current_price = self.price_history[symbol][-1]
        self.logger.log_analysis({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'price': current_price,
            'rsi': features['rsi_14'],
            'macd': features['macd'],
            'bb_position': features['bb_position'],
            'momentum': features['momentum_10'],
            'volatility': features['volatility'],
            'trend': features['trend'],
            'sentiment': ns,
            'recommendation': action,
            'confidence': confidence,
            'news_impact': nc
        })
        
        return action, confidence, reasoning

# ===================== Main Trading Bot Enhanced =====================

class FuturesTradingBot:
    """Enhanced main futures trading bot with comprehensive logging"""
    
    def __init__(self):
        self.config = FuturesConfig()
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        
        # Initialize logger
        self.logger = TradingLogger(LOG_FOLDER)
        
        # Initialize components with logger
        self.position_manager = FuturesPositionManager(
            self.config.INITIAL_BALANCE_USDT,
            self.config.LEVERAGE,
            self.logger
        )
        self.ai_predictor = FuturesAIPredictor(self.logger)
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.max_nav = self.config.INITIAL_BALANCE_USDT
        self.last_save_time = time.time()
        
        # Statistics
        self.stats = {
            'iterations': 0,
            'api_calls': 0,
            'errors': 0,
            'last_error': None
        }
    
    async def fetch_klines(self, symbol: str, interval: str = '30m', limit: int = 500):
        """Fetch historical klines with error handling"""
        try:
            self.stats['api_calls'] += 1
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            return [float(k[4]) for k in klines]  # Close prices
        except Exception as e:
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            self.logger.log_error(f"Error fetching klines for {symbol}: {e}")
            
            # Return last known prices if available
            if symbol in self.ai_predictor.price_history:
                return list(self.ai_predictor.price_history[symbol])[-limit:]
            return []
    
    async def run(self):
        """Main trading loop with enhanced logging"""
        self.logger.log_realtime("="*70)
        self.logger.log_realtime("🚀 Starting Enhanced Futures Trading Bot")
        self.logger.log_realtime(f"💰 Initial Balance: ${self.config.INITIAL_BALANCE_USDT:,.2f}")
        self.logger.log_realtime(f"📊 Leverage: {self.config.LEVERAGE}x")
        self.logger.log_realtime(f"🎯 Position Size: {self.config.POSITION_SIZE_PCT}% per trade")
        self.logger.log_realtime(f"📰 News Analysis: {'Enabled' if self.config.USE_NEWS_ANALYSIS else 'Disabled'}")
        self.logger.log_realtime("="*70)
        
        print("🚀 Starting Enhanced Futures Trading Bot")
        print(f"💰 Initial Balance: ${self.config.INITIAL_BALANCE_USDT:,.2f}")
        print(f"📊 Leverage: {self.config.LEVERAGE}x")
        print(f"🎯 Position Size: {self.config.POSITION_SIZE_PCT}% per trade")
        print(f"📰 News Analysis: {'Enabled' if self.config.USE_NEWS_ANALYSIS else 'Disabled'}")
        print(f"📁 Logs folder: {self.logger.date_folder}")
        print("="*70)
        
        # Initialize with historical data
        for symbol in self.config.SYMBOLS:
            prices = await self.fetch_klines(symbol)
            for price in prices:
                self.ai_predictor.update_price(symbol, price)
            print(f"✅ Loaded {len(prices)} historical prices for {symbol}")
            self.logger.log_realtime(f"Loaded {len(prices)} prices for {symbol}")
        
        while True:
            try:
                self.stats['iterations'] += 1
                iteration = self.stats['iterations']
                
                print(f"\n{'='*70}")
                print(f"⏱️ Iteration {iteration} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
                self.logger.log_realtime(f"\nIteration {iteration} started")
                
                # 1) Fetch current prices
                print("\n📡 Fetching current prices...")
                current_prices = {}
                
                for symbol in self.config.SYMBOLS:
                    try:
                        self.stats['api_calls'] += 1
                        ticker = self.client.futures_symbol_ticker(symbol=symbol)
                        price = float(ticker['price'])
                        current_prices[symbol] = price
                        self.ai_predictor.update_price(symbol, price)
                        
                        # Update position P&L
                        if symbol in self.position_manager.positions:
                            pnl = self.position_manager.update_position_pnl(symbol, price)
                            print(f"  {symbol}: ${price:,.2f} | Position P&L: ${pnl:+,.2f}")
                        else:
                            print(f"  {symbol}: ${price:,.2f}")
                            
                    except Exception as e:
                        self.stats['errors'] += 1
                        self.logger.log_error(f"Error fetching price for {symbol}: {e}")
                        # Use last known price
                        if self.ai_predictor.price_history[symbol]:
                            current_prices[symbol] = self.ai_predictor.price_history[symbol][-1]
                
                # 2) Check existing positions for exit
                for symbol in list(self.position_manager.positions.keys()):
                    if symbol in current_prices:
                        should_close, reason = self.position_manager.should_close_position(
                            symbol, current_prices[symbol]
                        )
                        
                        if should_close:
                            self.position_manager.close_position(symbol, current_prices[symbol], reason)
                
                # 3) Look for new opportunities
                if len(self.position_manager.positions) == 0:
                    print("\n🔍 Analyzing opportunities...")
                    self.logger.log_realtime("Analyzing market opportunities...")
                    
                    best_opportunity = None
                    best_confidence = 0
                    
                    for symbol in self.config.SYMBOLS:
                        
                        features = self.ai_predictor.calculate_features(symbol)
                        if not features:
                            print(f"  {symbol}: Insufficient data")
                            continue
                        
                        try:
                            action, confidence, reasoning = await self.ai_predictor.predict_with_news(
                                symbol, features
                            )
                            
                            print(f"  {symbol}: {action} (Confidence: {confidence:.1%}) - {reasoning}")
                            
                            if action != 'NEUTRAL' and confidence > best_confidence:
                                best_opportunity = (symbol, action, confidence, reasoning)
                                best_confidence = confidence
                                
                        except Exception as e:
                            self.logger.log_error(f"Error in prediction for {symbol}: {e}")
                            print(f"  {symbol}: Error in prediction")
                    
                    # Open best position
                    if best_opportunity and best_confidence > 0.6:
                        symbol, action, confidence, reasoning = best_opportunity
                        price = current_prices[symbol]
                        size = self.position_manager.calculate_position_size(
                            price, self.config.POSITION_SIZE_PCT
                        )
                        
                        self.position_manager.open_position(
                            symbol, action, price, size, reasoning, confidence
                        )
                
                # 4) Display and log portfolio status
                self._display_portfolio_status()
                
                # 5) Check risk limits
                drawdown = self._calculate_drawdown()
                if drawdown > self.config.MAX_DRAWDOWN_PCT:
                    print(f"\n🛑 MAX DRAWDOWN REACHED: {drawdown:.1f}%")
                    self.logger.log_realtime(f"MAX DRAWDOWN: {drawdown:.1f}% - Closing all positions")
                    
                    for symbol in list(self.position_manager.positions.keys()):
                        if symbol in current_prices:
                            self.position_manager.close_position(
                                symbol, current_prices[symbol], "MAX DRAWDOWN"
                            )
                
                # 6) Save logs periodically
                if time.time() - self.last_save_time > self.config.SAVE_INTERVAL:
                    self.logger.save_summary()
                    self.logger.create_daily_report()
                    self.last_save_time = time.time()
                    print("\n💾 Logs saved")
                
                # 7) Display statistics
                if iteration % 10 == 0:
                    self._display_statistics()
                
                await asyncio.sleep(self.config.UPDATE_INTERVAL)
                
            except KeyboardInterrupt:
                print("\n🛑 Shutting down...")
                self.logger.log_realtime("Bot stopped by user")
                self._display_final_results()
                break
                
            except Exception as e:
                self.stats['errors'] += 1
                self.logger.log_error(f"Main loop error: {str(e)}\n{traceback.format_exc()}")
                print(f"❌ Error in main loop: {e}")
                await asyncio.sleep(60)
    
    def _display_portfolio_status(self):
        """Display and log current portfolio status"""
        total_margin = 0
        total_pnl = 0
        
        for pos in self.position_manager.positions.values():
            margin = (pos['size'] * pos['entry_price']) / self.config.LEVERAGE
            total_margin += margin
            total_pnl += pos['unrealized_pnl']
        
        nav = self.position_manager.balance + total_margin + total_pnl
        self.max_nav = max(self.max_nav, nav)
        
        # Calculate metrics
        total_return_pct = ((nav / self.config.INITIAL_BALANCE_USDT) - 1) * 100
        drawdown_pct = ((self.max_nav - nav) / self.max_nav) * 100 if self.max_nav > 0 else 0
        
        # Calculate win rate
        if self.position_manager.position_history:
            wins = sum(1 for t in self.position_manager.position_history if t['pnl'] > 0)
            win_rate = (wins / len(self.position_manager.position_history)) * 100
        else:
            win_rate = 0
        
        # Calculate Sharpe ratio (simplified)
        if len(self.position_manager.position_history) > 1:
            returns = [t['pnl_pct'] for t in self.position_manager.position_history]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Log portfolio status
        self.logger.log_portfolio({
            'timestamp': datetime.now(),
            'free_balance': self.position_manager.balance,
            'used_margin': total_margin,
            'unrealized_pnl': total_pnl,
            'total_nav': nav,
            'drawdown_pct': drawdown_pct,
            'open_positions': len(self.position_manager.positions),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'total_return_pct': total_return_pct
        })
        
        # Display
        print(f"\n💼 Portfolio Status:")
        print(f"  Free Balance: ${self.position_manager.balance:,.2f}")
        print(f"  Used Margin: ${total_margin:,.2f}")
        print(f"  Unrealized P&L: ${total_pnl:+,.2f}")
        print(f"  Total NAV: ${nav:,.2f}")
        print(f"  Total Return: {total_return_pct:+.1f}%")
        print(f"  Drawdown: {drawdown_pct:.1f}%")
        print(f"  Open Positions: {len(self.position_manager.positions)}/{self.config.MAX_POSITIONS}")
        
        # Show position details
        if self.position_manager.positions:
            print("\n📊 Open Positions:")
            for symbol, pos in self.position_manager.positions.items():
                margin_used = (pos['size'] * pos['entry_price']) / self.config.LEVERAGE
                pnl_pct = (pos['unrealized_pnl'] / margin_used) * 100 if margin_used > 0 else 0
                hold_time = (datetime.utcnow() - pos['entry_time']).total_seconds() / 60
                
                emoji = "🟢" if pos['side'] == 'LONG' else "🔴"
                print(f"  {emoji} {pos['side']} {symbol}: {pos['size']:.4f} @ ${pos['entry_price']:,.2f}")
                print(f"     P&L: ${pos['unrealized_pnl']:+,.2f} ({pnl_pct:+.1f}%) | Hold: {hold_time:.0f}m")
                print(f"     SL: ${pos['stop_loss']:.2f} | TP: ${pos['take_profit']:.2f}")
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        total_margin = sum((pos['size'] * pos['entry_price']) / self.config.LEVERAGE 
                          for pos in self.position_manager.positions.values())
        total_pnl = sum(pos['unrealized_pnl'] for pos in self.position_manager.positions.values())
        current_nav = self.position_manager.balance + total_margin + total_pnl
        
        drawdown = ((self.max_nav - current_nav) / self.max_nav) * 100 if self.max_nav > 0 else 0
        return drawdown
    
    def _display_statistics(self):
        """Display trading statistics"""
        print("\n" + "="*70)
        print("📊 TRADING STATISTICS")
        print("="*70)
        
        runtime = (datetime.utcnow() - self.start_time).total_seconds() / 3600
        print(f"Runtime: {runtime:.1f} hours")
        print(f"Iterations: {self.stats['iterations']}")
        print(f"API Calls: {self.stats['api_calls']}")
        print(f"Errors: {self.stats['errors']}")
        
        if self.position_manager.position_history:
            total = len(self.position_manager.position_history)
            wins = sum(1 for t in self.position_manager.position_history if t['pnl'] > 0)
            
            print(f"\nTotal Trades: {total}")
            print(f"Win Rate: {(wins/total)*100:.1f}%")
            print(f"Total P&L: ${self.position_manager.total_pnl:,.2f}")
            print(f"Avg Trade: ${self.position_manager.total_pnl/total:,.2f}")
        
        print("="*70)
    
    def _display_final_results(self):
        """Display final trading results"""
        # Save final logs
        self.logger.save_summary()
        self.logger.create_daily_report()
        
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        runtime = (datetime.utcnow() - self.start_time).total_seconds() / 3600
        print(f"Runtime: {runtime:.1f} hours")
        
        if self.position_manager.position_history:
            total_trades = len(self.position_manager.position_history)
            winning_trades = sum(1 for t in self.position_manager.position_history if t['pnl'] > 0)
            
            print(f"\nTotal Trades: {total_trades}")
            print(f"Win Rate: {(winning_trades/total_trades)*100:.1f}%")
            print(f"Total P&L: ${self.position_manager.total_pnl:,.2f}")
            
            # Average metrics
            avg_pnl = self.position_manager.total_pnl / total_trades
            avg_hold = np.mean([t['duration_hours'] for t in self.position_manager.position_history])
            
            print(f"Average P&L per Trade: ${avg_pnl:,.2f}")
            print(f"Average Hold Time: {avg_hold:.1f} hours")
            
            # Best/Worst trades
            if total_trades > 0:
                best_trade = max(self.position_manager.position_history, key=lambda x: x['pnl'])
                worst_trade = min(self.position_manager.position_history, key=lambda x: x['pnl'])
                
                print(f"\nBest Trade: {best_trade['side']} {best_trade['symbol']} +${best_trade['pnl']:.2f} ({best_trade['pnl_pct']:+.1f}%)")
                print(f"Worst Trade: {worst_trade['side']} {worst_trade['symbol']} ${worst_trade['pnl']:.2f} ({worst_trade['pnl_pct']:.1f}%)")
        
        print(f"\nLog files saved in: {self.logger.date_folder}")
        print("="*70)

# ===================== Entry Point =====================

async def main():
    """Main entry point"""
    bot = FuturesTradingBot()
    await bot.run()

if __name__ == "__main__":
    # Create directories
    os.makedirs(LOG_FOLDER, exist_ok=True)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    # Run bot
    asyncio.run(main())
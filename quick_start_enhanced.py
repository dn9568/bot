# main_enhanced.py - Quick start for enhanced trading bot
# ========================================================
import os
import sys
import time
import csv
import pickle
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, deque

# Add modules path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import configs
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

# Import existing modules
from binance_client import create_binance_data_client, get_latest_prices
from executor import BinancePaperExecutor

# ===================== Enhanced AI Predictor =====================

class QuickEnhancedAI:
    """Simplified enhanced AI predictor using advanced techniques without heavy dependencies"""
    
    def __init__(self):
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        self.indicators_cache = defaultdict(dict)
        self.performance_tracker = defaultdict(lambda: {'wins': 0, 'losses': 0})
        self.min_history = 100
        
        # Load existing data
        self._load_history()
        
    def _load_history(self):
        """Load historical price data"""
        history_file = os.path.join(DATA_FOLDER, "price_history.pkl")
        if os.path.exists(history_file):
            try:
                with open(history_file, "rb") as f:
                    hist = pickle.load(f)
                for symbol, prices in hist.items():
                    for p in prices[-500:]:
                        self.price_history[symbol].append(p)
                print(f"✅ Loaded historical data for {len(hist)} symbols")
            except Exception as e:
                print(f"⚠️ Could not load history: {e}")
    
    def update_price(self, symbol: str, price: float):
        """Update price history"""
        self.price_history[symbol].append(price)
    
    def calculate_indicators(self, symbol: str) -> dict:
        """Calculate technical indicators without talib"""
        prices = list(self.price_history[symbol])
        if len(prices) < self.min_history:
            return {}
        
        prices_array = np.array(prices)
        indicators = {}
        
        # Simple Moving Averages
        indicators['sma_20'] = np.mean(prices[-20:])
        indicators['sma_50'] = np.mean(prices[-50:])
        indicators['sma_200'] = np.mean(prices[-200:]) if len(prices) >= 200 else indicators['sma_50']
        
        # Exponential Moving Averages
        indicators['ema_9'] = self._calculate_ema(prices_array, 9)
        indicators['ema_21'] = self._calculate_ema(prices_array, 21)
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(prices_array, 14)
        
        # MACD
        ema_12 = self._calculate_ema(prices_array, 12)
        ema_26 = self._calculate_ema(prices_array, 26)
        indicators['macd'] = ema_12 - ema_26
        indicators['macd_signal'] = self._calculate_ema(np.array([indicators['macd']]), 9)
        
        # Bollinger Bands
        bb_sma = indicators['sma_20']
        bb_std = np.std(prices[-20:])
        indicators['bb_upper'] = bb_sma + (2 * bb_std)
        indicators['bb_lower'] = bb_sma - (2 * bb_std)
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / bb_sma
        
        # Price position
        current_price = prices[-1]
        indicators['price_position'] = (current_price - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # Momentum
        indicators['momentum_10'] = (current_price - prices[-11]) / prices[-11] if len(prices) > 10 else 0
        indicators['momentum_20'] = (current_price - prices[-21]) / prices[-21] if len(prices) > 20 else 0
        
        # Volatility
        returns = np.diff(prices_array[-30:]) / prices_array[-31:-1]
        indicators['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Support/Resistance levels
        indicators['support'] = np.min(prices[-50:])
        indicators['resistance'] = np.max(prices[-50:])
        
        # Volume-based indicators (simulated)
        indicators['volume_trend'] = 1.0  # Placeholder
        
        # Market microstructure
        indicators['spread_pct'] = 0.001  # Simulated 0.1% spread
        
        self.indicators_cache[symbol] = indicators
        return indicators
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1]
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = np.abs(losses)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, symbol: str, indicators: dict) -> dict:
        """Generate trading signals from indicators"""
        signals = {
            'trend': 0,
            'momentum': 0,
            'mean_reversion': 0,
            'volatility': 0,
            'composite': 0
        }
        
        current_price = self.price_history[symbol][-1]
        
        # Trend following signals
        if current_price > indicators['sma_50'] > indicators['sma_200']:
            signals['trend'] = 1
        elif current_price < indicators['sma_50'] < indicators['sma_200']:
            signals['trend'] = -1
        
        # Momentum signals
        if indicators['macd'] > indicators['macd_signal'] and indicators['momentum_20'] > 0.02:
            signals['momentum'] = 1
        elif indicators['macd'] < indicators['macd_signal'] and indicators['momentum_20'] < -0.02:
            signals['momentum'] = -1
        
        # Mean reversion signals
        if indicators['rsi'] < 30 and indicators['price_position'] < 0.2:
            signals['mean_reversion'] = 1
        elif indicators['rsi'] > 70 and indicators['price_position'] > 0.8:
            signals['mean_reversion'] = -1
        
        # Volatility-based signals
        if indicators['volatility'] < 0.3:  # Low volatility
            if indicators['bb_width'] < 0.02:  # Bollinger squeeze
                signals['volatility'] = 1 if signals['trend'] > 0 else -1
        
        # Composite signal
        weights = {
            'trend': 0.3,
            'momentum': 0.3,
            'mean_reversion': 0.2,
            'volatility': 0.2
        }
        
        signals['composite'] = sum(signals[k] * weights[k] for k in weights)
        
        return signals
    
    def make_decision(self, symbol: str) -> tuple:
        """Make trading decision"""
        indicators = self.calculate_indicators(symbol)
        if not indicators:
            return 0, 0.0, "Insufficient data"
        
        signals = self.generate_signals(symbol, indicators)
        
        # Decision logic
        composite_score = signals['composite']
        
        # Calculate confidence based on signal agreement
        agreement = sum(1 for v in signals.values() if v * composite_score > 0) / len(signals)
        confidence = abs(composite_score) * agreement
        
        # Make decision
        if composite_score > 0.3 and confidence > 0.5:
            action = 1  # Buy
        elif composite_score < -0.3 and confidence > 0.5:
            action = -1  # Sell
        else:
            action = 0  # Hold
        
        # Generate reasoning
        reasons = []
        if indicators['rsi'] < 30:
            reasons.append(f"RSI oversold ({indicators['rsi']:.1f})")
        elif indicators['rsi'] > 70:
            reasons.append(f"RSI overbought ({indicators['rsi']:.1f})")
        
        if signals['trend'] != 0:
            reasons.append(f"{'Up' if signals['trend'] > 0 else 'Down'}trend detected")
        
        if abs(indicators['momentum_20']) > 0.05:
            reasons.append(f"Strong momentum ({indicators['momentum_20']:.1%})")
        
        reasoning = " | ".join(reasons[:3]) if reasons else "Mixed signals"
        
        return action, confidence, reasoning
    
    def select_best_opportunity(self, prices: dict) -> tuple:
        """Select best trading opportunity from multiple symbols"""
        opportunities = []
        
        for symbol, price in prices.items():
            self.update_price(symbol, price)
            
            if len(self.price_history[symbol]) < self.min_history:
                continue
            
            action, confidence, reasoning = self.make_decision(symbol)
            
            # Calculate score
            score = confidence * abs(action)
            
            # Adjust based on performance
            perf = self.performance_tracker[symbol]
            win_rate = perf['wins'] / (perf['wins'] + perf['losses']) if (perf['wins'] + perf['losses']) > 0 else 0.5
            score *= (0.5 + win_rate * 0.5)  # Boost score for winning symbols
            
            opportunities.append({
                'symbol': symbol,
                'action': action,
                'confidence': confidence,
                'score': score,
                'reasoning': reasoning
            })
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # Display analysis
        print("\n📊 Market Analysis:")
        print("-" * 60)
        for opp in opportunities[:5]:  # Show top 5
            action_str = "BUY 📈" if opp['action'] == 1 else "SELL 📉" if opp['action'] == -1 else "HOLD ⏸️"
            print(f"{opp['symbol']:<10} {action_str:<10} Score: {opp['score']:.3f}  Conf: {opp['confidence']:.1%}")
            print(f"           {opp['reasoning']}")
        
        if opportunities and opportunities[0]['score'] > 0:
            best = opportunities[0]
            return best['symbol'], best['action']
        
        return list(prices.keys())[0], 0

# ===================== Main Trading Loop =====================

def main():
    """Enhanced main trading loop"""
    print("🚀 Starting Enhanced AI Crypto Trading Bot")
    print("=" * 60)
    print(f"Initial Balance: ${INITIAL_BALANCE_USDT:,.2f}")
    print(f"Trade Amount: ${TRADE_AMOUNT_USDT:,.2f}")
    print(f"Target Pairs: {', '.join(TARGET_PAIRS)}")
    print("=" * 60)
    
    # Setup logging
    os.makedirs(LOG_FOLDER, exist_ok=True)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(LOG_FOLDER, "enhanced_bot.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    # CSV logging
    csv_path = os.path.join(LOG_FOLDER, "enhanced_trades.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "symbol", "action", "price", "amount",
                "balance", "position", "nav", "pnl", "total_return"
            ])
    
    # Initialize components
    print("\n🔧 Initializing components...")
    data_client = create_binance_data_client(BINANCE_API_KEY, BINANCE_API_SECRET)
    executor = BinancePaperExecutor(INITIAL_BALANCE_USDT, TRADE_AMOUNT_USDT)
    ai_predictor = QuickEnhancedAI()
    
    # Performance tracking
    start_balance = INITIAL_BALANCE_USDT
    max_nav = start_balance
    trades_count = 0
    winning_trades = 0
    
    print("✅ All systems ready!\n")
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            timestamp = datetime.utcnow()
            
            print(f"\n{'='*60}")
            print(f"🔄 Iteration {iteration} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            # 1) Fetch prices
            print("\n📡 Fetching live prices...")
            prices = get_latest_prices(data_client, TARGET_PAIRS)
            
            for sym, price in prices.items():
                print(f"  {sym}: ${price:,.2f}")
            
            # 2) AI Decision
            symbol, action = ai_predictor.select_best_opportunity(prices)
            trade_price = prices.get(symbol, 0.0)
            
            # 3) Execute trade
            prev_balance = executor.balance
            result = executor.execute(action, symbol, trade_price)
            
            # 4) Calculate metrics
            balance = result.get("balance", 0.0)
            position = result.get("position", 0.0)
            nav = balance + position * trade_price
            pnl = nav - max_nav
            total_return = ((nav / start_balance) - 1) * 100
            
            # Update tracking
            if action != 0:
                trades_count += 1
                if nav > max_nav:
                    winning_trades += 1
                    ai_predictor.performance_tracker[symbol]['wins'] += 1
                else:
                    ai_predictor.performance_tracker[symbol]['losses'] += 1
            
            max_nav = max(nav, max_nav)
            drawdown = ((max_nav - nav) / max_nav) * 100 if max_nav > 0 else 0
            
            # 5) Display status
            print(f"\n💼 Portfolio Status:")
            print(f"  Balance: ${balance:,.2f}")
            print(f"  Position: {position:.6f} {symbol if position > 0 else ''}")
            print(f"  Position Value: ${position * trade_price:,.2f}")
            print(f"  NAV: ${nav:,.2f}")
            print(f"  P&L: ${pnl:,.2f}")
            print(f"  Total Return: {total_return:+.2f}%")
            print(f"  Drawdown: {drawdown:.2f}%")
            
            if trades_count > 0:
                win_rate = (winning_trades / trades_count) * 100
                print(f"  Win Rate: {win_rate:.1f}% ({winning_trades}/{trades_count})")
            
            # 6) Log to CSV
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp.isoformat(),
                    symbol,
                    "BUY" if action == 1 else "SELL" if action == -1 else "HOLD",
                    f"{trade_price:.2f}",
                    TRADE_AMOUNT_USDT,
                    f"{balance:.2f}",
                    f"{position:.6f}",
                    f"{nav:.2f}",
                    f"{pnl:.2f}",
                    f"{total_return:.2f}"
                ])
            
            # 7) Save state periodically
            if iteration % 10 == 0:
                history_file = os.path.join(DATA_FOLDER, "price_history.pkl")
                with open(history_file, "wb") as f:
                    pickle.dump(dict(ai_predictor.price_history), f)
                print(f"\n💾 Saved price history")
            
            # 8) Risk check
            if drawdown > 20:
                print("\n⚠️ WARNING: Drawdown exceeds 20%! Consider reducing position size.")
            
            if nav < start_balance * 0.8:
                print("\n🛑 STOP LOSS: Portfolio down 20%. Consider stopping.")
            
        except KeyboardInterrupt:
            print("\n\n🛑 Trading bot stopped by user")
            print(f"Final NAV: ${nav:,.2f}")
            print(f"Total Return: {total_return:+.2f}%")
            break
            
        except Exception as e:
            logging.error(f"Error in main loop: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
            time.sleep(60)
            continue
        
        # Dynamic sleep
        if ai_predictor.indicators_cache.get(symbol, {}).get('volatility', 0) > 0.5:
            sleep_time = 30  # High volatility = more frequent updates
        else:
            sleep_time = SLEEP_INTERVAL
        
        print(f"\n⏳ Next update in {sleep_time}s...")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
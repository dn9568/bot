# integration_layer.py - Integration with existing system
# =========================================================
import os
import sys
import time
import pickle
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Tuple

# Add modules path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import existing modules
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

# Import new advanced system
from advanced_trading_bot import (
    TradingConfig,
    AITradingEngine,
    RiskManager,
    MarketDataManager,
    AdvancedIndicators,
    FeatureEngineer
)

class EnhancedAIDecider:
    """Enhanced AI Decider that combines existing and new AI systems"""
    
    def __init__(self):
        # Initialize advanced AI engine
        self.config = TradingConfig(
            binance_api_key=BINANCE_API_KEY,
            binance_api_secret=BINANCE_API_SECRET,
            initial_balance=INITIAL_BALANCE_USDT,
            base_trade_amount=TRADE_AMOUNT_USDT,
            confidence_threshold=0.6,
            paper_trading=True
        )
        
        self.ai_engine = AITradingEngine(self.config)
        self.risk_manager = RiskManager(self.config)
        self.indicators = AdvancedIndicators()
        
        # Load existing price history if available
        self._load_existing_data()
        
    def _load_existing_data(self):
        """Load existing price history from pickle files"""
        history_file = os.path.join(DATA_FOLDER, "price_history.pkl")
        if os.path.exists(history_file):
            try:
                with open(history_file, "rb") as f:
                    hist = pickle.load(f)
                for symbol, prices in hist.items():
                    for price in prices[-500:]:  # Load last 500 prices
                        self.ai_engine.update_market_data(symbol, {
                            'close': price,
                            'high': price * 1.001,
                            'low': price * 0.999,
                            'open': price,
                            'volume': 1000000
                        })
                print(f"✅ Loaded historical data for {len(hist)} symbols")
            except Exception as e:
                print(f"❌ Error loading history: {e}")
    
    def update_price_history(self, symbol: str, price: float):
        """Update price history for both systems"""
        # Update new AI engine
        self.ai_engine.update_market_data(symbol, {
            'close': price,
            'high': price * 1.001,  # Approximate high/low
            'low': price * 0.999,
            'open': price,
            'volume': 1000000  # Default volume
        })
    
    def select_and_decide(self, prices: Dict[str, float]) -> Tuple[str, int]:
        """Enhanced decision making combining multiple AI models"""
        best_symbol = None
        best_action = 0
        best_score = 0
        
        analysis_results = []
        
        for symbol, price in prices.items():
            # Update price
            self.update_price_history(symbol, price)
            
            # Get prediction from advanced AI
            action, confidence, reasoning = self.ai_engine.predict(symbol)
            
            # Calculate additional metrics
            market_data = list(self.ai_engine.market_data[symbol])
            if len(market_data) >= 20:
                # Calculate quick indicators
                prices_array = np.array([d['close'] for d in market_data])
                
                rsi = self.indicators.rsi(prices_array)[-1]
                ema_9 = self.indicators.ema(prices_array, 9)[-1]
                ema_21 = self.indicators.ema(prices_array, 21)[-1]
                
                # Momentum score
                momentum = (price - prices_array[-20]) / prices_array[-20]
                
                # Volume analysis (simulated)
                volume_ratio = 1.0  # Default
                
                # Combined score
                score = confidence
                
                # Adjust score based on indicators
                if action == 1:  # Buy signal
                    if rsi < 30:
                        score *= 1.2  # Oversold bonus
                    if price > ema_9 > ema_21:
                        score *= 1.1  # Uptrend bonus
                elif action == -1:  # Sell signal
                    if rsi > 70:
                        score *= 1.2  # Overbought bonus
                    if price < ema_9 < ema_21:
                        score *= 1.1  # Downtrend bonus
                
                # Risk adjustment
                if not self.risk_manager.check_risk_limits(symbol, action, TRADE_AMOUNT_USDT):
                    score *= 0.5
                
                analysis_results.append({
                    'symbol': symbol,
                    'action': action,
                    'confidence': confidence,
                    'score': score,
                    'rsi': rsi,
                    'momentum': momentum,
                    'reasoning': reasoning
                })
                
                if score > best_score and action != 0:
                    best_symbol = symbol
                    best_action = action
                    best_score = score
        
        # Log analysis
        print("\n📊 Market Analysis:")
        for result in sorted(analysis_results, key=lambda x: x['score'], reverse=True):
            action_str = "BUY" if result['action'] == 1 else "SELL" if result['action'] == -1 else "HOLD"
            print(f"{result['symbol']}: {action_str} (Score: {result['score']:.3f}, "
                  f"RSI: {result['rsi']:.1f}, Momentum: {result['momentum']:.1%})")
        
        if best_symbol:
            print(f"\n🎯 Selected: {best_symbol} - {'BUY' if best_action == 1 else 'SELL'}")
            return best_symbol, best_action
        
        # Default to hold on first symbol
        return list(prices.keys())[0], 0

# ===================== Enhanced Main Loop =====================

def run_enhanced_bot():
    """Run enhanced trading bot with existing infrastructure"""
    print("🚀 Starting Enhanced AI Trading Bot")
    print("="*50)
    
    # Create components
    data_client = create_binance_data_client(BINANCE_API_KEY, BINANCE_API_SECRET)
    executor = BinancePaperExecutor(INITIAL_BALANCE_USDT, TRADE_AMOUNT_USDT)
    ai_decider = EnhancedAIDecider()
    
    # CSV logging setup
    os.makedirs(LOG_FOLDER, exist_ok=True)
    csv_path = os.path.join(LOG_FOLDER, "enhanced_bot_log.csv")
    
    # Performance tracking
    performance = {
        'start_balance': INITIAL_BALANCE_USDT,
        'trades': [],
        'equity_curve': []
    }
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            print(f"\n⏱️ Iteration {iteration} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            # 1) Fetch real prices
            prices = get_latest_prices(data_client, TARGET_PAIRS)
            print("\n💹 Current Prices:")
            for sym, price in prices.items():
                print(f"  {sym}: ${price:,.2f}")
            
            # 2) Enhanced AI Decision
            symbol, action = ai_decider.select_and_decide(prices)
            trade_price = prices.get(symbol, 0.0)
            
            # 3) Execute trade
            result = executor.execute(action, symbol, trade_price)
            
            # 4) Calculate performance
            balance = result.get("balance", 0.0)
            position = result.get("position", 0.0)
            nav = balance + position * trade_price
            
            # Track performance
            performance['equity_curve'].append({
                'timestamp': datetime.utcnow(),
                'nav': nav,
                'balance': balance,
                'position_value': position * trade_price
            })
            
            if action != 0:
                performance['trades'].append({
                    'timestamp': datetime.utcnow(),
                    'symbol': symbol,
                    'action': 'BUY' if action == 1 else 'SELL',
                    'price': trade_price,
                    'nav': nav
                })
            
            # 5) Display status
            print(f"\n💼 Portfolio Status:")
            print(f"  Balance: ${balance:,.2f}")
            print(f"  Position Value: ${position * trade_price:,.2f}")
            print(f"  NAV: ${nav:,.2f}")
            print(f"  Total Return: {((nav / INITIAL_BALANCE_USDT) - 1) * 100:.2f}%")
            
            # 6) Risk metrics
            if len(performance['equity_curve']) > 1:
                equity_values = [e['nav'] for e in performance['equity_curve']]
                max_equity = max(equity_values)
                current_drawdown = (max_equity - nav) / max_equity * 100
                print(f"  Current Drawdown: {current_drawdown:.2f}%")
                
                # Calculate Sharpe ratio
                if len(equity_values) > 30:
                    returns = np.diff(equity_values) / equity_values[:-1]
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0
                    print(f"  Sharpe Ratio: {sharpe:.2f}")
            
            # Save state
            if iteration % 10 == 0:
                save_performance_report(performance)
            
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            save_performance_report(performance)
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logging.error(f"Error in main loop: {e}", exc_info=True)
        
        # Dynamic sleep based on market conditions
        sleep_time = calculate_dynamic_interval(ai_decider, prices)
        print(f"\n⏳ Sleeping {sleep_time}s...")
        time.sleep(sleep_time)

def calculate_dynamic_interval(ai_decider, prices):
    """Calculate dynamic sleep interval based on market volatility"""
    # Get volatility from recent price movements
    volatilities = []
    
    for symbol in prices:
        market_data = list(ai_decider.ai_engine.market_data[symbol])
        if len(market_data) >= 20:
            prices_array = np.array([d['close'] for d in market_data[-20:]])
            returns = np.diff(prices_array) / prices_array[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            volatilities.append(volatility)
    
    if volatilities:
        avg_volatility = np.mean(volatilities)
        
        # High volatility = shorter intervals
        if avg_volatility > 0.5:  # 50% annualized vol
            return 30
        elif avg_volatility > 0.3:  # 30% annualized vol
            return 45
        else:
            return SLEEP_INTERVAL
    
    return SLEEP_INTERVAL

def save_performance_report(performance):
    """Save detailed performance report"""
    report_path = os.path.join(LOG_FOLDER, "performance_report.json")
    
    # Calculate metrics
    equity_values = [e['nav'] for e in performance['equity_curve']]
    
    if len(equity_values) > 1:
        total_return = (equity_values[-1] / equity_values[0] - 1) * 100
        max_drawdown = calculate_max_drawdown(equity_values)
        
        # Win rate
        winning_trades = sum(1 for i, t in enumerate(performance['trades']) 
                           if i > 0 and t['nav'] > performance['trades'][i-1]['nav'])
        win_rate = winning_trades / len(performance['trades']) * 100 if performance['trades'] else 0
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len(performance['trades']),
            'win_rate_pct': win_rate,
            'final_nav': equity_values[-1],
            'runtime_hours': (datetime.utcnow() - performance['equity_curve'][0]['timestamp']).total_seconds() / 3600
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Performance report saved to {report_path}")

def calculate_max_drawdown(equity_values):
    """Calculate maximum drawdown"""
    peak = equity_values[0]
    max_dd = 0
    
    for value in equity_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_dd:
            max_dd = drawdown
    
    return max_dd

# ===================== Utility Functions =====================

def backtest_enhanced_system(historical_data_path: str):
    """Backtest the enhanced system on historical data"""
    print("🔄 Starting backtest...")
    
    # Load historical data
    with open(historical_data_path, 'rb') as f:
        historical_data = pickle.load(f)
    
    # Initialize components
    ai_decider = EnhancedAIDecider()
    executor = BinancePaperExecutor(INITIAL_BALANCE_USDT, TRADE_AMOUNT_USDT)
    
    results = []
    
    # Simulate trading
    for timestamp, prices in historical_data.items():
        symbol, action = ai_decider.select_and_decide(prices)
        trade_price = prices.get(symbol, 0.0)
        result = executor.execute(action, symbol, trade_price)
        
        results.append({
            'timestamp': timestamp,
            'action': action,
            'symbol': symbol,
            'nav': result['balance'] + result['position'] * trade_price
        })
    
    # Calculate performance
    initial_nav = INITIAL_BALANCE_USDT
    final_nav = results[-1]['nav']
    total_return = (final_nav / initial_nav - 1) * 100
    
    print(f"\n📈 Backtest Results:")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Total Trades: {sum(1 for r in results if r['action'] != 0)}")
    
    return results

# ===================== Advanced Features =====================

class MarketRegimeDetector:
    """Detect market regimes for adaptive strategies"""
    
    def __init__(self):
        self.regimes = {
            'bull': {'threshold': 0.02, 'lookback': 50},
            'bear': {'threshold': -0.02, 'lookback': 50},
            'ranging': {'threshold': 0.01, 'lookback': 20},
            'volatile': {'threshold': 0.03, 'lookback': 10}
        }
    
    def detect_regime(self, prices: np.ndarray) -> str:
        """Detect current market regime"""
        if len(prices) < 50:
            return 'unknown'
        
        # Calculate metrics
        returns = np.diff(prices) / prices[:-1]
        
        # Trend
        trend = (prices[-1] - prices[-50]) / prices[-50]
        
        # Volatility
        volatility = np.std(returns[-20:])
        
        # Regime detection
        if trend > self.regimes['bull']['threshold']:
            return 'bull' if volatility < 0.02 else 'volatile_bull'
        elif trend < self.regimes['bear']['threshold']:
            return 'bear' if volatility < 0.02 else 'volatile_bear'
        elif volatility > self.regimes['volatile']['threshold']:
            return 'volatile_ranging'
        else:
            return 'ranging'

class PortfolioOptimizer:
    """Optimize portfolio allocation across multiple assets"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def optimize_weights(self, returns: pd.DataFrame, 
                        method: str = 'sharpe') -> Dict[str, float]:
        """Calculate optimal portfolio weights"""
        if method == 'sharpe':
            return self._maximize_sharpe(returns)
        elif method == 'min_variance':
            return self._minimize_variance(returns)
        elif method == 'risk_parity':
            return self._risk_parity(returns)
        else:
            # Equal weight
            symbols = returns.columns
            return {sym: 1.0/len(symbols) for sym in symbols}
    
    def _maximize_sharpe(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Maximize Sharpe ratio"""
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Simple mean-variance optimization
        # In practice, use scipy.optimize
        weights = (mean_returns - self.risk_free_rate) / returns.std()
        weights = weights / weights.sum()
        
        return dict(zip(returns.columns, weights))
    
    def _minimize_variance(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Minimize portfolio variance"""
        # Simplified - equal weight for now
        symbols = returns.columns
        return {sym: 1.0/len(symbols) for sym in symbols}
    
    def _risk_parity(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Risk parity allocation"""
        # Allocate inversely proportional to volatility
        volatilities = returns.std()
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        return dict(zip(returns.columns, weights))

# ===================== Entry Points =====================

def main():
    """Main entry point for enhanced bot"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced AI Trading Bot')
    parser.add_argument('--mode', choices=['live', 'backtest', 'train'], 
                       default='live', help='Operating mode')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--data', type=str, help='Historical data path for backtest')
    
    args = parser.parse_args()
    
    if args.mode == 'live':
        run_enhanced_bot()
    elif args.mode == 'backtest':
        if args.data:
            backtest_enhanced_system(args.data)
        else:
            print("❌ Please provide historical data path for backtest")
    elif args.mode == 'train':
        print("🎓 Training mode - Coming soon!")

if __name__ == "__main__":
    # Import numpy for calculations
    import numpy as np
    import pandas as pd
    import logging
    
    main()
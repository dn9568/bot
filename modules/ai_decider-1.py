# advanced_ai_decider.py - Smart AI Trading Engine
# -------------------------------------------------------------------
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
import time
from typing import Dict, List, Tuple, Optional
import logging

class AdvancedAI:
    def __init__(self):
        self.price_history = {}
        self.trade_history = []
        self.market_sentiment = 0.0
        self.volatility_scores = {}
        self.momentum_scores = {}
        self.support_resistance = {}
        self.fear_greed_index = 50
        
    def update_price_history(self, prices: Dict[str, float]):
        """อัพเดตประวัติราคาสำหรับการวิเคราะห์"""
        timestamp = datetime.now()
        
        for symbol, price in prices.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append({
                'timestamp': timestamp,
                'price': price
            })
            
            # เก็บข้อมูล 200 รอบล่าสุด
            if len(self.price_history[symbol]) > 200:
                self.price_history[symbol] = self.price_history[symbol][-200:]
    
    def calculate_technical_indicators(self, symbol: str) -> Dict:
        """คำนวณตัวชี้วัดทางเทคนิค"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return {}
        
        prices = [item['price'] for item in self.price_history[symbol]]
        df = pd.DataFrame({'price': prices})
        
        indicators = {}
        
        # Moving Averages
        indicators['sma_10'] = df['price'].rolling(window=10).mean().iloc[-1] if len(prices) >= 10 else prices[-1]
        indicators['sma_20'] = df['price'].rolling(window=20).mean().iloc[-1] if len(prices) >= 20 else prices[-1]
        indicators['ema_12'] = df['price'].ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = df['price'].ewm(span=26).mean().iloc[-1]
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1])) if len(prices) >= 14 else 50
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = pd.Series([indicators['macd']]).ewm(span=9).mean().iloc[0]
        
        # Bollinger Bands
        sma_20 = df['price'].rolling(window=20).mean()
        std_20 = df['price'].rolling(window=20).std()
        indicators['bb_upper'] = (sma_20 + 2 * std_20).iloc[-1] if len(prices) >= 20 else prices[-1] * 1.02
        indicators['bb_lower'] = (sma_20 - 2 * std_20).iloc[-1] if len(prices) >= 20 else prices[-1] * 0.98
        
        # Volatility
        indicators['volatility'] = df['price'].pct_change().rolling(window=20).std().iloc[-1] * 100 if len(prices) >= 20 else 1.0
        
        # Price momentum
        if len(prices) >= 10:
            indicators['momentum_5'] = (prices[-1] - prices[-6]) / prices[-6] * 100
            indicators['momentum_10'] = (prices[-1] - prices[-11]) / prices[-11] * 100
        else:
            indicators['momentum_5'] = 0
            indicators['momentum_10'] = 0
        
        return indicators
    
    def get_market_sentiment(self) -> float:
        """วิเคราะห์ความรู้สึกของตลาด"""
        try:
            # Fear & Greed Index (จำลอง)
            # ในการใช้งานจริงสามารถใช้ API เช่น Alternative.me
            import random
            self.fear_greed_index = random.randint(0, 100)
            
            if self.fear_greed_index < 25:
                sentiment = -0.8  # Extreme Fear = Good buy opportunity
            elif self.fear_greed_index < 45:
                sentiment = -0.4  # Fear = Moderate buy
            elif self.fear_greed_index < 55:
                sentiment = 0.0   # Neutral
            elif self.fear_greed_index < 75:
                sentiment = 0.4   # Greed = Moderate sell
            else:
                sentiment = 0.8   # Extreme Greed = Good sell opportunity
            
            return sentiment
        except:
            return 0.0
    
    def analyze_volume_pattern(self, symbol: str) -> float:
        """วิเคราะห์รูปแบบปริมาณการซื้อขาย"""
        # จำลองการวิเคราะห์ปริมาณ
        import random
        volume_score = random.uniform(-1, 1)
        return volume_score
    
    def find_support_resistance(self, symbol: str) -> Dict:
        """หาจุด Support และ Resistance"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 50:
            return {'support': 0, 'resistance': 0}
        
        prices = [item['price'] for item in self.price_history[symbol]]
        current_price = prices[-1]
        
        # หา Support และ Resistance แบบง่าย
        recent_prices = prices[-50:]
        support = min(recent_prices) * 1.001  # Support + 0.1%
        resistance = max(recent_prices) * 0.999  # Resistance - 0.1%
        
        return {
            'support': support,
            'resistance': resistance,
            'distance_to_support': (current_price - support) / support * 100,
            'distance_to_resistance': (resistance - current_price) / current_price * 100
        }
    
    def calculate_risk_reward(self, symbol: str, action: int, price: float) -> float:
        """คำนวณอัตราส่วน Risk:Reward"""
        sr = self.find_support_resistance(symbol)
        
        if action == 1:  # Buy
            target = sr['resistance']
            stop_loss = sr['support']
            potential_profit = (target - price) / price * 100
            potential_loss = (price - stop_loss) / price * 100
        elif action == -1:  # Sell
            target = sr['support']
            stop_loss = sr['resistance']
            potential_profit = (price - target) / price * 100
            potential_loss = (stop_loss - price) / price * 100
        else:
            return 0
        
        if potential_loss > 0:
            risk_reward = potential_profit / potential_loss
        else:
            risk_reward = 0
        
        return risk_reward
    
    def multi_timeframe_analysis(self, symbol: str) -> Dict:
        """วิเคราะห์หลายไทม์เฟรม"""
        if symbol not in self.price_history:
            return {'short_term': 0, 'medium_term': 0, 'long_term': 0}
        
        prices = [item['price'] for item in self.price_history[symbol]]
        
        analysis = {}
        
        # Short-term (5 periods)
        if len(prices) >= 5:
            short_change = (prices[-1] - prices[-5]) / prices[-5] * 100
            analysis['short_term'] = 1 if short_change > 1 else -1 if short_change < -1 else 0
        else:
            analysis['short_term'] = 0
        
        # Medium-term (20 periods)
        if len(prices) >= 20:
            medium_change = (prices[-1] - prices[-20]) / prices[-20] * 100
            analysis['medium_term'] = 1 if medium_change > 2 else -1 if medium_change < -2 else 0
        else:
            analysis['medium_term'] = 0
        
        # Long-term (50 periods)
        if len(prices) >= 50:
            long_change = (prices[-1] - prices[-50]) / prices[-50] * 100
            analysis['long_term'] = 1 if long_change > 5 else -1 if long_change < -5 else 0
        else:
            analysis['long_term'] = 0
        
        return analysis
    
    def advanced_decision(self, prices: Dict[str, float]) -> Tuple[str, int, Dict]:
        """ตัดสินใจการเทรดแบบอัจฉริยะ"""
        
        # อัพเดตประวัติราคา
        self.update_price_history(prices)
        
        # วิเคราะห์ความรู้สึกตลาด
        market_sentiment = self.get_market_sentiment()
        
        best_symbol = None
        best_action = 0
        best_score = -999
        analysis_results = {}
        
        print(f"🧠 Advanced AI Analysis (Market Sentiment: {market_sentiment:.2f}, F&G Index: {self.fear_greed_index})")
        
        for symbol in prices.keys():
            try:
                # คำนวณตัวชี้วัดทางเทคนิค
                indicators = self.calculate_technical_indicators(symbol)
                
                if not indicators:
                    continue
                
                # วิเคราะห์หลายไทม์เฟรม
                timeframe_analysis = self.multi_timeframe_analysis(symbol)
                
                # หา Support/Resistance
                sr = self.find_support_resistance(symbol)
                
                # คำนวณคะแนนรวม
                score = 0
                reasons = []
                
                # 1. Technical Indicators (40% weight)
                current_price = prices[symbol]
                
                # RSI Score
                rsi = indicators.get('rsi', 50)
                if rsi < 30:
                    score += 20  # Oversold = Buy signal
                    reasons.append(f"RSI Oversold ({rsi:.1f})")
                elif rsi > 70:
                    score -= 20  # Overbought = Sell signal
                    reasons.append(f"RSI Overbought ({rsi:.1f})")
                
                # Moving Average Score
                sma_10 = indicators.get('sma_10', current_price)
                sma_20 = indicators.get('sma_20', current_price)
                if current_price > sma_10 > sma_20:
                    score += 15
                    reasons.append("Price above MAs")
                elif current_price < sma_10 < sma_20:
                    score -= 15
                    reasons.append("Price below MAs")
                
                # MACD Score
                macd = indicators.get('macd', 0)
                macd_signal = indicators.get('macd_signal', 0)
                if macd > macd_signal and macd > 0:
                    score += 10
                    reasons.append("MACD Bullish")
                elif macd < macd_signal and macd < 0:
                    score -= 10
                    reasons.append("MACD Bearish")
                
                # 2. Multi-timeframe Analysis (25% weight)
                tf_score = (timeframe_analysis['short_term'] * 0.5 + 
                           timeframe_analysis['medium_term'] * 0.3 + 
                           timeframe_analysis['long_term'] * 0.2) * 25
                score += tf_score
                if tf_score != 0:
                    reasons.append(f"Multi-TF: {tf_score:.1f}")
                
                # 3. Support/Resistance (20% weight)
                if sr['support'] > 0 and sr['resistance'] > 0:
                    if sr['distance_to_support'] < 2:  # Near support
                        score += 15
                        reasons.append("Near Support")
                    elif sr['distance_to_resistance'] < 2:  # Near resistance
                        score -= 15
                        reasons.append("Near Resistance")
                
                # 4. Market Sentiment (15% weight)
                sentiment_score = market_sentiment * 15
                score += sentiment_score
                if abs(sentiment_score) > 5:
                    reasons.append(f"Market Sentiment: {sentiment_score:.1f}")
                
                # 5. Volatility Filter
                volatility = indicators.get('volatility', 1)
                if volatility > 5:  # High volatility
                    score *= 0.7  # Reduce confidence
                    reasons.append("High Volatility")
                
                # กำหนด Action
                if score > 25:
                    action = 1  # Buy
                elif score < -25:
                    action = -1  # Sell
                else:
                    action = 0  # Hold
                
                # คำนวณ Risk:Reward
                risk_reward = self.calculate_risk_reward(symbol, action, current_price)
                
                # ปรับคะแนนตาม Risk:Reward
                if risk_reward > 2:  # Good risk:reward ratio
                    score += 10
                    reasons.append(f"Good R:R ({risk_reward:.1f})")
                elif risk_reward < 1 and action != 0:
                    score -= 20
                    reasons.append(f"Poor R:R ({risk_reward:.1f})")
                
                # บันทึกผลการวิเคราะห์
                analysis_results[symbol] = {
                    'score': score,
                    'action': action,
                    'reasons': reasons,
                    'indicators': indicators,
                    'risk_reward': risk_reward,
                    'support_resistance': sr
                }
                
                print(f"📊 {symbol}: Score={score:.1f}, Action={'BUY' if action==1 else 'SELL' if action==-1 else 'HOLD'}")
                print(f"   Reasons: {', '.join(reasons[:3])}")
                
                # เลือกคู่เงินที่ดีที่สุด
                if abs(score) > abs(best_score):
                    best_score = score
                    best_symbol = symbol
                    best_action = action
                    
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {e}")
                continue
        
        # ตรวจสอบความเชื่อมั่นในการตัดสินใจ
        if abs(best_score) < 20:
            best_action = 0  # Hold if not confident enough
            print(f"🤔 Low confidence ({best_score:.1f}), switching to HOLD")
        
        if best_symbol is None:
            try:
                from config import TARGET_PAIRS
                best_symbol = TARGET_PAIRS[0]["symbol"]
            except:
                best_symbol = "BTCUSDT"
            best_action = 0
        
        print(f"🎯 Final Decision: {best_symbol} -> {'BUY' if best_action==1 else 'SELL' if best_action==-1 else 'HOLD'} (Score: {best_score:.1f})")
        
        return best_symbol, best_action, analysis_results

# Global AI instance
advanced_ai = AdvancedAI()

def ai_decide(prices: dict) -> tuple:
    """Main AI decision function with advanced analysis"""
    try:
        symbol, action, analysis = advanced_ai.advanced_decision(prices)
        return symbol, action
    except Exception as e:
        print(f"❌ Advanced AI failed: {e}")
        # Fallback to simple strategy
        import random
        try:
            from config import TARGET_PAIRS
            symbols = [p["symbol"] for p in TARGET_PAIRS]
        except:
            symbols = list(prices.keys())
        
        symbol = random.choice(symbols)
        action = random.choice([-1, 0, 1])
        return symbol, action


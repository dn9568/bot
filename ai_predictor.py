# ai_predictor.py - Advanced AI Prediction System for Live Trading
# -------------------------------------------------------------------
import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime, timedelta
import talib
import warnings
warnings.filterwarnings('ignore')

class AdvancedAIPredictor:
    def __init__(self, model_path='ai_models/advanced_trading_ai'):
        self.model_path = model_path
        self.models = {}
        self.meta_model = None
        self.scaler = None
        self.feature_cols = None
        self.price_history = {}
        self.prediction_history = []
        self.confidence_threshold = 0.65
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            print("📂 Loading AI models...")
            # Load individual models
            model_files = ['xgboost', 'lightgbm', 'random_forest',
                          'gradient_boost', 'neural_network']
            for model_name in model_files:
                try:
                    with open(f'{self.model_path}/{model_name}_model.pkl', 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                        print(f"✅ Loaded {model_name} model")
                except:
                    print(f"⚠️ Could not load {model_name} model")
            # Load meta-model
            with open(f'{self.model_path}/meta_model.pkl', 'rb') as f:
                self.meta_model = pickle.load(f)
                print("✅ Loaded meta-model")
            # Load scaler
            with open(f'{self.model_path}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
                print("✅ Loaded feature scaler")
            # Load feature columns
            with open(f'{self.model_path}/features.json', 'r') as f:
                self.feature_cols = json.load(f)
                print(f"✅ Loaded {len(self.feature_cols)} features")
            print("🎯 All models loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def update_price_history(self, symbol, price, volume=None):
        """Update price history for feature calculation"""
        if symbol not in self.price_history:
            self.price_history[symbol] = {
                'prices': [],
                'volumes': [],
                'timestamps': []
            }
        self.price_history[symbol]['prices'].append(price)
        self.price_history[symbol]['volumes'].append(volume if volume else 1000000)
        self.price_history[symbol]['timestamps'].append(datetime.now())
        # Keep last 500 data points
        if len(self.price_history[symbol]['prices']) > 500:
            self.price_history[symbol]['prices'] = self.price_history[symbol]['prices'][-500:]
            self.price_history[symbol]['volumes'] = self.price_history[symbol]['volumes'][-500:]
            self.price_history[symbol]['timestamps'] = self.price_history[symbol]['timestamps'][-500:]
    
    def calculate_live_features(self, symbol, current_price):
        """Calculate features from live data"""
        if symbol not in self.price_history:
            return None
        prices = self.price_history[symbol]['prices']
        volumes = self.price_history[symbol]['volumes']
        if len(prices) < 200:
            return None
        # Create temporary DataFrame
        df = pd.DataFrame({
            'Close': prices,
            'Volume': volumes,
            'High': pd.Series(prices).rolling(5).max(),
            'Low': pd.Series(prices).rolling(5).min(),
            'Open': pd.Series(prices).shift(1)
        })
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        # Calculate features
        features = {}
        # Price-based
        features['returns'] = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1)
        features['log_returns'] = np.log(df['Close'].iloc[-1] / df['Close'].iloc[-2])
        features['price_range'] = (df['High'].iloc[-1] - df['Low'].iloc[-1]) / df['Close'].iloc[-1]
        features['close_to_high'] = (df['High'].iloc[-1] - df['Close'].iloc[-1]) / df['High'].iloc[-1]
        features['close_to_low'] = (df['Close'].iloc[-1] - df['Low'].iloc[-1]) / df['Low'].iloc[-1]
        # Volume
        vol_mean_20 = df['Volume'].rolling(20).mean().iloc[-1]
        features['volume_ratio'] = df['Volume'].iloc[-1] / vol_mean_20 if vol_mean_20 > 0 else 1
        vol_mean_5 = df['Volume'].rolling(5).mean().iloc[-1]
        features['volume_trend'] = vol_mean_5 / vol_mean_20 if vol_mean_20 > 0 else 1
        # Indicators arrays
        close_array = np.array(df['Close'])
        high_array = np.array(df['High'])
        low_array = np.array(df['Low'])
        volume_array = np.array(df['Volume'])
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(close_array) >= period:
                sma = talib.SMA(close_array, timeperiod=period)[-1]
                ema = talib.EMA(close_array, timeperiod=period)[-1]
                features[f'SMA_{period}'] = sma
                features[f'EMA_{period}'] = ema
                features[f'SMA_ratio_{period}'] = current_price / sma if sma > 0 else 1
            else:
                features[f'SMA_{period}'] = current_price
                features[f'EMA_{period}'] = current_price
                features[f'SMA_ratio_{period}'] = 1
        # RSI
        features['RSI_14'] = talib.RSI(close_array, timeperiod=14)[-1] if len(close_array) >= 14 else 50
        features['RSI_28'] = talib.RSI(close_array, timeperiod=28)[-1] if len(close_array) >= 28 else 50
        # MACD
        macd, signal, hist = talib.MACD(close_array)
        features['MACD'] = macd[-1] if not np.isnan(macd[-1]) else 0
        features['MACD_signal'] = signal[-1] if not np.isnan(signal[-1]) else 0
        features['MACD_hist'] = hist[-1] if not np.isnan(hist[-1]) else 0
        # Fill any missing features
        for col in self.feature_cols or []:
            if col not in features:
                features[col] = 0
        return features
    
    def analyze_decision(self, features, prediction, confidence):
        """Analyze and explain the trading decision"""
        reasons = []
        rsi = features.get('RSI_14', 50)
        if rsi < 30:
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            reasons.append(f"RSI overbought ({rsi:.1f})")
        macd_hist = features.get('MACD_hist', 0)
        if macd_hist > 0:
            reasons.append("MACD bullish")
        elif macd_hist < 0:
            reasons.append("MACD bearish")
        sma_ratio_20 = features.get('SMA_ratio_20', 1)
        if sma_ratio_20 > 1.02:
            reasons.append("Price above SMA20")
        elif sma_ratio_20 < 0.98:
            reasons.append("Price below SMA20")
        vol_ratio = features.get('volume_ratio', 1)
        if vol_ratio > 1.5:
            reasons.append("High volume")
        reasons.append(f"AI confidence {confidence:.1%}")
        return ", ".join(reasons[:3])
    
    def get_market_regime(self, symbol):
        """Identify current market regime"""
        if symbol not in self.price_history:
            return "Unknown"
        prices = self.price_history[symbol]['prices']
        if len(prices) < 50:
            return "Unknown"
        returns = pd.Series(prices).pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] * 100
        sma_50 = pd.Series(prices).rolling(50).mean().iloc[-1]
        current_price = prices[-1]
        trend_strength = (current_price - sma_50) / sma_50 * 100
        if volatility > 5:
            return "Volatile Trending" if abs(trend_strength) > 5 else "Volatile Ranging"
        else:
            return "Stable Trending" if abs(trend_strength) > 3 else "Stable Ranging"
    
    def adjust_for_regime(self, action, confidence, regime):
        """Adjust trading decision based on market regime"""
        if "Volatile" in regime and confidence < 0.75:
            return 0
        if "Ranging" in regime and action != 0:
            confidence *= 0.7
        return action
    
    def get_recommendation(self, prices_dict):
        """Get trading recommendation for multiple symbols"""
        recommendations = []
        for symbol, price in prices_dict.items():
            action, confidence, reason = self.predict(symbol, price)
            regime = self.get_market_regime(symbol)
            adj_action = self.adjust_for_regime(action, confidence, regime)
            score = confidence * abs(adj_action)
            recommendations.append({
                'symbol': symbol,
                'action': adj_action,
                'confidence': confidence,
                'score': score,
                'reason': reason,
                'regime': regime,
                'price': price
            })
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        if recommendations and recommendations[0]['score'] > 0:
            best = recommendations[0]
            print(f"\n🎯 AI Recommendation:")
            print(f"Symbol: {best['symbol']}")
            print(f"Action: {'BUY' if best['action'] == 1 else 'SELL' if best['action'] == -1 else 'HOLD'}")
            print(f"Confidence: {best['confidence']:.1%}")
            print(f"Regime: {best['regime']}")
            print(f"Reason: {best['reason']}")
            return best['symbol'], best['action']
        return list(prices_dict.keys())[0], 0

    def predict(self, symbol, current_price, volume=None):
        """Make prediction using ensemble models"""
        self.update_price_history(symbol, current_price, volume)
        features = self.calculate_live_features(symbol, current_price)
        if features is None:
            print("⚠️ Not enough historical data for prediction")
            return 0, 0.5, "Insufficient data"
        X = pd.DataFrame([features])[self.feature_cols]
        X = X.fillna(0)
        X_scaled = self.scaler.transform(X)
        predictions, probabilities = [], []
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)[0]; predictions.append(pred)
                if hasattr(model, 'predict_proba'):
                    probabilities.append(model.predict_proba(X_scaled)[0])
            except:
                pass
        if probabilities:
            meta_features = np.hstack(probabilities).reshape(1, -1)
            meta_pred = self.meta_model.predict(meta_features)[0]
            meta_proba = self.meta_model.predict_proba(meta_features)[0]
            if meta_pred <= 1:
                action, confidence = -1, max(meta_proba[0], meta_proba[1])
            elif meta_pred == 2:
                action, confidence = 0, meta_proba[2]
            else:
                action, confidence = 1, max(meta_proba[3], meta_proba[4])
        else:
            votes = {-1:0, 0:0, 1:0}
            for p in predictions:
                votes[-1 if p<=1 else 0 if p==2 else 1] += 1
            action = max(votes, key=votes.get)
            confidence = votes[action] / len(predictions) if predictions else 0
        reason = self.analyze_decision(features, meta_pred if probabilities else None, confidence)
        self.prediction_history.append({
            'timestamp': datetime.now(), 'symbol': symbol,
            'price': current_price, 'action': action,
            'confidence': confidence, 'reason': reason
        })
        return action, confidence, reason

# Integration with existing ai_decide function
_ai_predictor = None

def advanced_ai_decide(prices: dict) -> tuple:
    """Advanced AI decision function"""
    global _ai_predictor
    try:
        if _ai_predictor is None:
            _ai_predictor = AdvancedAIPredictor()
        symbol, action = _ai_predictor.get_recommendation(prices)
        return symbol, action
    except Exception as e:
        print(f"❌ Advanced AI error: {e}")
        return list(prices.keys())[0], 0

# For backward compatibility
ai_decide = advanced_ai_decide

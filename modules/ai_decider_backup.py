# modules/ai_decider.py
# -------------------------------------------------------------------
import os
import json
import pickle
import numpy as np
import pandas as pd
import talib
from warnings import filterwarnings

filterwarnings('ignore')

class AdvancedAIPredictor:
    """
    คลาสสำหรับโหลดโมเดล AI ที่เทรนแล้ว, จัดการข้อมูลราคา,
    และทำการตัดสินใจซื้อขาย
    """
    def __init__(self, model_path='ai_models/advanced_trading_ai'):
        self.models = {}
        self.meta_model = None
        self.scaler = None
        self.feature_cols = None
        self.model_path = model_path
        self.history = {}
        self.min_history_size = 210

        try:
            self._load_models()
            print("🎯 All models loaded successfully!")
        except Exception as e:
            print(f"❌ Critical error loading models: {e}")
            raise

    def _load_models(self):
        """โหลดโมเดล, scaler, และรายชื่อ features จากไฟล์"""
        print("📂 Loading AI models...")
        model_names = ['xgboost', 'lightgbm', 'random_forest', 'gradient_boost', 'neural_network']
        for name in model_names:
            with open(os.path.join(self.model_path, f'{name}_model.pkl'), 'rb') as f:
                self.models[name] = pickle.load(f)
                print(f"✅ Loaded {name} model")

        with open(os.path.join(self.model_path, 'meta_model.pkl'), 'rb') as f:
            self.meta_model = pickle.load(f)
            print("✅ Loaded meta-model")

        with open(os.path.join(self.model_path, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
            print("✅ Loaded feature scaler")

        with open(os.path.join(self.model_path, 'features.json'), 'r') as f:
            self.feature_cols = json.load(f)
            print(f"✅ Loaded {len(self.feature_cols)} features")

    def update_price_history(self, symbol, price):
        """อัปเดตข้อมูลราคาย้อนหลังสำหรับเหรียญที่กำหนด"""
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(price)
        self.history[symbol] = self.history[symbol][-self.min_history_size - 50:]

    def _calculate_features(self, symbol):
        """คำนวณ features ทั้งหมดจากข้อมูลย้อนหลังของเหรียญที่กำหนด"""
        prices = self.history.get(symbol)
        if not prices or len(prices) < self.min_history_size:
            print(f"⚠️ Not enough historical data for {symbol} prediction ({len(prices)}/{self.min_history_size} points)")
            return None
        
        # --- หมายเหตุสำคัญ ---
        # ข้อมูลที่เรามีคือราคาปิด (Close) เท่านั้น แต่ Indicators จำนวนมากต้องการ
        # ราคาเปิด (Open), สูงสุด (High), ต่ำสุด (Low) ด้วย
        # โค้ดส่วนนี้จะจำลองค่า O, H, L โดยใช้ราคา C เพื่อให้ talib ทำงานได้
        # ซึ่งอาจทำให้ค่า Indicator บางตัวไม่แม่นยำ 100% แต่เพียงพอให้โค้ดรันได้
        df = pd.DataFrame(prices, columns=['Close'])
        df['Open'] = df['Close'].shift(1).fillna(df['Close'])
        df['High'] = df[['Open', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Close']].min(axis=1)
        df['Volume'] = 1_000_000 

        # === START FEATURE CALCULATION (เพิ่มส่วนที่ขาดไปทั้งหมด) ===

        # Basic Features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility and Range
        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        df['spread'] = df['High'] - df['Low']
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_60'] = df['returns'].rolling(60).std()
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_60']
        df['TRANGE'] = talib.TRANGE(df['High'], df['Low'], df['Close'])
        df['ATR_14'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

        # Volume-based
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)

        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = talib.SMA(df['Close'], timeperiod=period)
            df[f'EMA_{period}'] = talib.EMA(df['Close'], timeperiod=period)
            df[f'SMA_ratio_{period}'] = df['Close'] / df[f'SMA_{period}']

        # Oscillators
        df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
        df['RSI_28'] = talib.RSI(df['Close'], timeperiod=28)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'])
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
        df['STOCHF_K'], df['STOCHF_D'] = talib.STOCHF(df['High'], df['Low'], df['Close'])
        df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'])
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)

        # Momentum
        df['MOM_10'] = talib.MOM(df['Close'], timeperiod=10)
        df['ROC_10'] = talib.ROC(df['Close'], timeperiod=10)
        df['price_acceleration'] = df['returns'].diff()

        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'])
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Other Price Transformations
        df['typical_price'] = talib.TYPPRICE(df['High'], df['Low'], df['Close'])
        df['weighted_close'] = talib.WCLPRICE(df['High'], df['Low'], df['Close'])
        df['close_to_high'] = (df['High'] - df['Close']) / df['High']
        df['close_to_low'] = (df['Close'] - df['Low']) / df['Low']

        # Lagged Features
        for lag in range(1, 11):
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)

        # Rolling Statistics
        for period in [5, 10, 20]:
            df[f'return_mean_{period}'] = df['returns'].rolling(period).mean()
            df[f'return_std_{period}'] = df['returns'].rolling(period).std()
            df[f'return_skew_{period}'] = df['returns'].rolling(period).skew()
            df[f'return_kurt_{period}'] = df['returns'].rolling(period).kurt()
        
        # Candlestick Patterns (will be 0 if H=L=C)
        df['DOJI'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        df['HAMMER'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        df['ENGULFING'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
        
        # Volume Price Trend (approximation)
        df['volume_price_trend'] = (df['returns'] * df['Volume']).cumsum()
        
        # === END FEATURE CALCULATION ===
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # เลือกเฉพาะ features ที่โมเดลต้องการ และเอาแถวล่าสุด
        last_features = df[self.feature_cols].iloc[-1:]
        
        # เติมค่าที่หายไป (ถ้ามี) ด้วยค่าก่อนหน้า หรือ 0
        last_features.fillna(method='ffill', inplace=True)
        last_features.fillna(0, inplace=True)
        
        if last_features.empty or last_features.isnull().values.any():
             print(f"⚠️ Feature calculation for {symbol} resulted in NaN values. Skipping.")
             return None

        return last_features

    def select_and_decide(self, current_prices):
        """วิเคราะห์ทุกเหรียญที่มีข้อมูลพอ และเลือกเหรียญที่ดีที่สุดเพื่อทำการซื้อขาย"""
        best_opportunity = {'symbol': 'BTCUSDT', 'action': 0, 'confidence': 0.0}
        
        for symbol in current_prices.keys():
            features = self._calculate_features(symbol)
            
            if features is None:
                continue

            X_scaled = self.scaler.transform(features)

            base_predictions = []
            for model in self.models.values():
                pred_proba = model.predict_proba(X_scaled)
                base_predictions.append(pred_proba)
            
            X_meta = np.hstack(base_predictions)
            meta_pred = self.meta_model.predict(X_meta)[0]
            meta_pred_proba = self.meta_model.predict_proba(X_meta)[0]

            confidence = 0.0
            action = 0
            if meta_pred == 4: # Strong Buy
                action = 1
                confidence = meta_pred_proba[4]
            elif meta_pred == 0: # Strong Sell
                action = -1
                confidence = meta_pred_proba[0]

            print(f"🔬 Analysis for {symbol}: Prediction={meta_pred}, Confidence={confidence:.2f}, Action={action}")

            if confidence > best_opportunity['confidence'] and confidence > 0.5:
                best_opportunity.update({
                    'symbol': symbol,
                    'action': action,
                    'confidence': confidence
                })

        print(f"🏆 Best opportunity: {best_opportunity['symbol']} with action {best_opportunity['action']} (Confidence: {best_opportunity['confidence']:.2f})")
        return best_opportunity['symbol'], best_opportunity['action']
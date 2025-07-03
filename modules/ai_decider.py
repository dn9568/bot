# modules/ai_decider.py

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from config import DECISION_THRESHOLD_BUY, DECISION_THRESHOLD_SELL

warnings.filterwarnings('ignore')

class AdvancedAIPredictor:
    def __init__(self):
        # สร้าง path แบบ absolute จากโฟลเดอร์โปรเจกต์
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.model_path = os.path.join(base_dir, 'ai_models', 'advanced_trading_ai')
        print(f"📂 Looking for models in {self.model_path}")

        self.models = {}
        self.meta_model = None
        self.scaler = None
        self.feature_cols = []
        self.history = defaultdict(list)
        self.min_history_size = 210

        # โหลดโมเดล, scaler, features
        self._load_models()

    def _load_models(self):
        print("📂 Loading AI models...")
        model_names = ['xgboost','lightgbm','random_forest','gradient_boost','neural_network']
        for name in model_names:
            file_path = os.path.join(self.model_path, f'{name}_model.pkl')
            if not os.path.exists(file_path):
                print(f"❌ Missing model file: {file_path}")
                continue
            try:
                with open(file_path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"✅ Loaded {name}")
            except Exception as e:
                print(f"❌ Could not load {name}: {e}")

        # meta-model
        meta_path = os.path.join(self.model_path, 'meta_model.pkl')
        if not os.path.exists(meta_path):
            print(f"❌ Missing meta-model file: {meta_path}")
        else:
            try:
                with open(meta_path, 'rb') as f:
                    self.meta_model = pickle.load(f)
                print("✅ Loaded meta-model")
            except Exception as e:
                print(f"❌ Could not load meta-model: {e}")

        # scaler
        scaler_path = os.path.join(self.model_path, 'scaler.pkl')
        if not os.path.exists(scaler_path):
            print(f"❌ Missing scaler file: {scaler_path}")
        else:
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✅ Loaded scaler")
            except Exception as e:
                print(f"❌ Could not load scaler: {e}")

        # features.json
        feat_path = os.path.join(self.model_path, 'features.json')
        if not os.path.exists(feat_path):
            print(f"❌ Missing features.json: {feat_path}")
        else:
            try:
                with open(feat_path, 'r', encoding='utf-8') as f:
                    self.feature_cols = json.load(f)
                print(f"✅ Loaded {len(self.feature_cols)} features")
            except Exception as e:
                print(f"❌ Could not load features.json: {e}")

        print("🎯 Models ready!")

    def update_price_history(self, symbol, price):
        clean = symbol.replace('/', '')
        self.history[clean].append(float(price))
        if len(self.history[clean]) > 1000:
            self.history[clean] = self.history[clean][-1000:]

    def _calculate_features(self, sym):
        prices = None
        for v in (sym, sym.replace('/','')):
            if v in self.history:
                prices = self.history[v]
                break
        if not prices or len(prices) < self.min_history_size:
            print(f"⚠️ Not enough data for {sym} ({len(prices) if prices else 0}/{self.min_history_size})")
            return None

        df = pd.DataFrame({'Close': prices})
        feats = {
            'returns':        df.Close.iloc[-1]/df.Close.iloc[-2]-1,
            'returns_5':      df.Close.iloc[-1]/df.Close.iloc[-6]-1 if len(df)>5 else 0,
            'returns_20':     df.Close.iloc[-1]/df.Close.iloc[-21]-1 if len(df)>20 else 0,
            'sma_5':          df.Close.tail(5).mean(),
            'sma_20':         df.Close.tail(20).mean(),
            'sma_50':         df.Close.tail(50).mean() if len(df)>50 else df.Close.tail(20).mean(),
            'price_to_sma5':  df.Close.iloc[-1]/df.Close.tail(5).mean(),
            'price_to_sma20': df.Close.iloc[-1]/df.Close.tail(20).mean(),
            'volatility_5':   df.Close.tail(5).std(),
            'volatility_20':  df.Close.tail(20).std()
        }
        row = pd.DataFrame([feats])
        for col in self.feature_cols:
            if col not in row:
                row[col] = 0
        row = row[self.feature_cols]
        return row.values  # shape (1,n)

    def select_and_decide(self, current_prices):
        best_symbol = None
        best_action = 0
        best_conf = 0.0

        for sym, price in current_prices.items():
            clean = sym.replace('/','')
            X = self._calculate_features(clean)
            if X is None:
                continue

            Xs = self.scaler.transform(X) if self.scaler else X
            probs = []
            for m in self.models.values():
                try:
                    probs.append(m.predict_proba(Xs)[0][1])
                except:
                    probs.append(float(m.predict(Xs)[0]))
            avg = np.mean(probs)
            final = 1 if avg>=0.5 else 0
            conf = abs(avg-0.5)*2

            if final==1 and conf>=DECISION_THRESHOLD_BUY:
                action=1
            elif final==0 and conf>=DECISION_THRESHOLD_SELL:
                action=-1
            else:
                action=0

            if action!=0 and conf>best_conf:
                best_symbol, best_action, best_conf = clean, action, conf

        if not best_symbol:
            best_symbol, best_action = list(current_prices.keys())[0].replace('/',''), 0

        print(f"🏆 Best opportunity: {best_symbol} with action {best_action} (Confidence: {best_conf:.2f})")
        return best_symbol, best_action

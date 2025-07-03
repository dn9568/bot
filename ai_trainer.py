# ai_trainer.py - Advanced AI Model Training for Futures Trading
# -------------------------------------------------------------------
import os
import json
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import talib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')


class AdvancedAITrainer:
    def __init__(self, data_path='./data'):
        self.data_path = data_path
        self.performance_metrics = {}

    def load_data(self, symbol='BTC'):
        """Load and prepare historical data from CSV in data/"""
        path = os.path.join(self.data_path, f'{symbol}_yfinance_daily.csv')
        if not os.path.exists(path):
            print(f"❌ ไม่พบไฟล์: {path}")
            return None

        try:
            df = pd.read_csv(
                path,
                skiprows=3,
                header=None,
                names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
            )
            # Convert types
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')

            df.sort_values('Date', inplace=True)
            df.set_index('Date', inplace=True)
            df.dropna(inplace=True)

            print(f"✅ Loaded {len(df)} rows of {symbol} data from {path}")
            return df

        except Exception as e:
            print(f"❌ Error loading data from {path}: {e}")
            return None

    def calculate_advanced_features(self, df):
        """Calculate comprehensive technical indicators"""
        print("📊 Calculating advanced technical features...")
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        df['close_to_high'] = (df['High'] - df['Close']) / df['High']
        df['close_to_low'] = (df['Close'] - df['Low']) / df['Low']
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()

        # Moving averages, momentum, patterns, etc.
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = talib.SMA(df['Close'], timeperiod=period)
            df[f'EMA_{period}'] = talib.EMA(df['Close'], timeperiod=period)
            df[f'SMA_ratio_{period}'] = df['Close'] / df[f'SMA_{period}']

        df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
        df['RSI_28'] = talib.RSI(df['Close'], timeperiod=28)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'])
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'])
        df['ATR_14'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)

        df.dropna(inplace=True)
        return df

    def create_labels(self, df, lookahead=1, threshold=0.001):
        """Create trading labels based on future returns"""
        df['future_return'] = df['Close'].shift(-lookahead) / df['Close'] - 1
        conditions = [
            df['future_return'] < -threshold * 2,
            (df['future_return'] >= -threshold * 2) & (df['future_return'] < -threshold),
            (df['future_return'] >= -threshold) & (df['future_return'] <= threshold),
            (df['future_return'] > threshold) & (df['future_return'] <= threshold * 2),
            df['future_return'] > threshold * 2
        ]
        choices = [0, 1, 2, 3, 4]
        df['label'] = np.select(conditions, choices, default=2)
        df['binary_label'] = (df['future_return'] > threshold).astype(int)
        df.dropna(inplace=True)
        return df

    def prepare_features(self, df):
        """Prepare and select features for training"""
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'future_return', 'label', 'binary_label', 'symbol']
        cols = [c for c in df.columns if c not in exclude]
        valid = [c for c in cols if df[c].isna().mean() < 0.1]
        return valid

    def train_ensemble_models(self, X_train, y_train, X_val, y_val):
        print("🚀 Training ensemble models...")
        models = {
            'xgboost': xgb.XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.01,
                                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                                         use_label_encoder=False, eval_metric='mlogloss'),
            'lightgbm': lgb.LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.01,
                                           subsample=0.8, colsample_bytree=0.8, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=200, max_depth=15,
                                                   min_samples_split=10, min_samples_leaf=5,
                                                   random_state=42, n_jobs=-1),
            'gradient_boost': GradientBoostingClassifier(n_estimators=200, max_depth=8,
                                                         learning_rate=0.01, subsample=0.8,
                                                         random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100,50,25), activation='relu',
                                            solver='adam', alpha=0.001, learning_rate='adaptive',
                                            max_iter=500, random_state=42)
        }
        trained, perf = {}, {}
        for name, model in models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                f1  = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                perf[name] = {'accuracy': acc, 'f1': f1}
                trained[name] = model
                print(f"✅ {name} - Acc: {acc:.4f}, F1: {f1:.4f}")
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
        return trained, perf

    def create_meta_model(self, models, X_train, y_train, X_val, y_val):
        print("🔮 Creating meta-model ensemble...")
        metas, vals = [], []
        for mdl in models.values():
            try:
                metas.append(mdl.predict_proba(X_train))
                vals.append(mdl.predict_proba(X_val))
            except:
                metas.append(mdl.predict(X_train).reshape(-1,1))
                vals.append(mdl.predict(X_val).reshape(-1,1))
        Xm_train = np.hstack(metas)
        Xm_val = np.hstack(vals)
        meta = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.01,
                                  random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        meta.fit(Xm_train, y_train)
        y_pred = meta.predict(Xm_val)
        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        print(f"✅ Meta-model - Acc: {acc:.4f}, F1: {f1:.4f}")
        return meta

    def save_models(self, models, meta_model, scaler, feature_cols):
        print("💾 Saving models...")
        out = 'ai_models/advanced_trading_ai'
        os.makedirs(out, exist_ok=True)
        for name, m in models.items():
            with open(f'{out}/{name}_model.pkl', 'wb') as f:
                pickle.dump(m, f)
        with open(f'{out}/meta_model.pkl', 'wb') as f:
            pickle.dump(meta_model, f)
        with open(f'{out}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open(f'{out}/features.json', 'w') as f:
            json.dump(feature_cols, f)
        info = {'created_at': datetime.now().isoformat(), 'models': list(models.keys()),
                'feature_count': len(feature_cols), 'performance': self.performance_metrics}
        with open(f'{out}/model_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        print("✅ Models saved successfully!")

    def train_complete_system(self):
        print("🚀 Starting Advanced AI Training System")
        print("="*50)
        data_list = []
        for sym in ['BTC', 'LTC']:
            print(f"\n📈 Processing {sym} data...")
            df = self.load_data(sym)
            if df is None:
                continue
            df = self.calculate_advanced_features(df)
            df = self.create_labels(df, lookahead=1, threshold=0.002)
            df['symbol'] = sym
            data_list.append(df)
        combined = pd.concat(data_list).sort_index().dropna()
        print(f"\n📊 Total samples: {len(combined)}")
        feature_cols = self.prepare_features(combined)
        print(f"📐 Selected {len(feature_cols)} features")
        X = combined[feature_cols]
        y = combined['label']
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=5)
        all_models, all_perf = [], []
        for i, (tr, va) in enumerate(tscv.split(X_scaled), 1):
            print(f"\n📁 Training Fold {i}/5")
            Xt, Xv = X_scaled[tr], X_scaled[va]
            yt, yv = y.iloc[tr], y.iloc[va]
            ms, pf = self.train_ensemble_models(Xt, yt, Xv, yv)
            all_models.append(ms)
            all_perf.append(pf)
        best = all_models[-1]
        self.performance_metrics = all_perf[-1]
        Xt, Xv = X_scaled[tr], X_scaled[va]
        yt, yv = y.iloc[tr], y.iloc[va]
        meta = self.create_meta_model(best, Xt, yt, Xv, yv)
        self.save_models(best, meta, scaler, feature_cols)
        print("\n" + "="*50)
        print("🎉 Training completed successfully!")
        print("📁 Models saved in: ai_models/advanced_trading_ai/")
        return True


def main():
    """Main entry point: initialize AI predictor with historical warm-up, then start paper trading loop"""
    # 1) Initialize predictor
    predictor = AdvancedAIPredictor()
    
    # 2) Warm up historical price history from CSVs
    symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'LTC']
    for sym in symbols:
        csv_path = os.path.join('data', f'{sym}_yfinance_daily.csv')
        if not os.path.exists(csv_path):
            continue
        # Read close prices
        hist_df = pd.read_csv(
            csv_path,
            skiprows=3,
            header=None,
            names=['Date','Close','High','Low','Open','Volume'],
            usecols=['Close']
        )
        # Feed into predictor
        for price in hist_df['Close'].tolist():
            predictor.update_price_history(sym, price)

    # 3) Start the paper trading loop (existing code)
    print("🚀 Using Advanced AI Trading System")
    main_logic(predictor)

if __name__ == '__main__':
    main()
    main()

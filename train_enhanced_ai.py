# train_enhanced_ai.py - Training script for enhanced AI models
# =============================================================
import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add modules path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# For basic training (no heavy dependencies)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ===================== Simple Trainer (ไม่ต้องติดตั้งอะไรเพิ่ม) =====================

class SimpleAITrainer:
    """Simple AI trainer using only sklearn - works with existing dependencies"""
    
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.models = {}
        self.scaler = RobustScaler()
        self.feature_names = []
        
    def load_all_data(self):
        """Load data from CSV files"""
        print("📂 Loading historical data...")
        
        all_data = []
        symbols = ['BTC', 'ETH', 'LTC', 'BNB']
        
        for symbol in symbols:
            csv_file = os.path.join(self.data_path, f'{symbol}_yfinance_daily.csv')
            if not os.path.exists(csv_file):
                print(f"⚠️ Skipping {symbol} - file not found")
                continue
                
            try:
                # Read CSV with proper header handling
                df = pd.read_csv(csv_file, skiprows=3)
                df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
                
                # Convert to numeric
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna()
                df = df.sort_values('Date')
                df['Symbol'] = symbol
                
                print(f"✅ Loaded {len(df)} rows for {symbol}")
                all_data.append(df)
                
            except Exception as e:
                print(f"❌ Error loading {symbol}: {e}")
        
        if not all_data:
            raise ValueError("No data loaded!")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n📊 Total data points: {len(combined_df)}")
        
        return combined_df
    
    def create_features(self, df):
        """Create technical features"""
        print("\n🔧 Creating features...")
        
        features = pd.DataFrame()
        
        # Group by symbol for feature calculation
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date')
            
            # Price features
            symbol_data['returns'] = symbol_data['Close'].pct_change()
            symbol_data['log_returns'] = np.log(symbol_data['Close'] / symbol_data['Close'].shift(1))
            symbol_data['price_range'] = (symbol_data['High'] - symbol_data['Low']) / symbol_data['Close']
            symbol_data['close_position'] = (symbol_data['Close'] - symbol_data['Low']) / (symbol_data['High'] - symbol_data['Low'])
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                symbol_data[f'sma_{period}'] = symbol_data['Close'].rolling(period).mean()
                symbol_data[f'sma_ratio_{period}'] = symbol_data['Close'] / symbol_data[f'sma_{period}']
            
            # RSI
            symbol_data['rsi_14'] = self.calculate_rsi(symbol_data['Close'], 14)
            
            # MACD
            ema_12 = symbol_data['Close'].ewm(span=12).mean()
            ema_26 = symbol_data['Close'].ewm(span=26).mean()
            symbol_data['macd'] = ema_12 - ema_26
            symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9).mean()
            symbol_data['macd_hist'] = symbol_data['macd'] - symbol_data['macd_signal']
            
            # Bollinger Bands
            sma_20 = symbol_data['Close'].rolling(20).mean()
            std_20 = symbol_data['Close'].rolling(20).std()
            symbol_data['bb_upper'] = sma_20 + (std_20 * 2)
            symbol_data['bb_lower'] = sma_20 - (std_20 * 2)
            symbol_data['bb_width'] = (symbol_data['bb_upper'] - symbol_data['bb_lower']) / sma_20
            symbol_data['bb_position'] = (symbol_data['Close'] - symbol_data['bb_lower']) / (symbol_data['bb_upper'] - symbol_data['bb_lower'])
            
            # Volume features
            symbol_data['volume_ratio'] = symbol_data['Volume'] / symbol_data['Volume'].rolling(20).mean()
            
            # Volatility
            symbol_data['volatility_20'] = symbol_data['returns'].rolling(20).std()
            
            # Momentum
            symbol_data['momentum_10'] = symbol_data['Close'].pct_change(10)
            symbol_data['momentum_20'] = symbol_data['Close'].pct_change(20)
            
            features = pd.concat([features, symbol_data], ignore_index=True)
        
        # Create target variable (next day return)
        features['target_return'] = features.groupby('Symbol')['Close'].pct_change().shift(-1)
        
        # Create labels: -1 (sell), 0 (hold), 1 (buy)
        features['label'] = 0
        features.loc[features['target_return'] > 0.002, 'label'] = 1  # Buy if > 0.2%
        features.loc[features['target_return'] < -0.002, 'label'] = -1  # Sell if < -0.2%
        
        # Drop NaN values
        features = features.dropna()
        
        print(f"✅ Created {len(features.columns)} features")
        print(f"📊 Label distribution:")
        print(features['label'].value_counts())
        
        return features
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_models(self, features_df):
        """Train machine learning models"""
        print("\n🚀 Training models...")
        
        # Select feature columns
        exclude_cols = ['Date', 'Symbol', 'target_return', 'label', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        X = features_df[feature_cols].values
        y = features_df['label'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Train each model
        best_scores = {}
        
        for name, model in models.items():
            print(f"\n📈 Training {name}...")
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Balance classes by adjusting -1 to 0 and 1 to 2 for sklearn
                y_train_adj = y_train + 1
                y_val_adj = y_val + 1
                
                model.fit(X_train, y_train_adj)
                y_pred = model.predict(X_val)
                
                accuracy = accuracy_score(y_val_adj, y_pred)
                scores.append(accuracy)
                print(f"  Fold {fold+1}: Accuracy = {accuracy:.4f}")
            
            avg_score = np.mean(scores)
            best_scores[name] = avg_score
            self.models[name] = model
            print(f"  Average accuracy: {avg_score:.4f}")
        
        print("\n✅ Training completed!")
        return best_scores
    
    def save_models(self):
        """Save trained models and scaler"""
        model_dir = 'ai_models/simple_enhanced'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_path = os.path.join(model_dir, f'{name}_model.pkl')
            joblib.dump(model, model_path)
            print(f"💾 Saved {name} to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"💾 Saved scaler to {scaler_path}")
        
        # Save feature names
        features_path = os.path.join(model_dir, 'features.json')
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
        print(f"💾 Saved features to {features_path}")
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'models': list(self.models.keys()),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names[:10] + ['...']  # Show first 10
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ All models saved to {model_dir}/")
        print(f"📋 Total features: {len(self.feature_names)}")

# ===================== Advanced Trainer (ต้องติดตั้ง xgboost, lightgbm) =====================

class AdvancedAITrainer(SimpleAITrainer):
    """Advanced trainer with XGBoost and LightGBM"""
    
    def __init__(self, data_path='data'):
        super().__init__(data_path)
        self.advanced_models = {}
        
    def train_advanced_models(self, features_df):
        """Train advanced models (requires xgboost, lightgbm)"""
        try:
            import xgboost as xgb
            import lightgbm as lgb
            
            print("\n🚀 Training advanced models...")
            
            # Prepare data
            exclude_cols = ['Date', 'Symbol', 'target_return', 'label', 'Open', 'High', 'Low', 'Close', 'Volume']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            X = features_df[feature_cols].values
            y = features_df['label'].values + 1  # Adjust labels for sklearn
            
            X_scaled = self.scaler.transform(X)
            
            # XGBoost
            print("\n📈 Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.01,
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                use_label_encoder=False
            )
            xgb_model.fit(X_scaled, y)
            self.advanced_models['xgboost'] = xgb_model
            
            # LightGBM
            print("📈 Training LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.01,
                objective='multiclass',
                num_class=3,
                random_state=42
            )
            lgb_model.fit(X_scaled, y)
            self.advanced_models['lightgbm'] = lgb_model
            
            print("✅ Advanced models trained!")
            
        except ImportError:
            print("⚠️ XGBoost/LightGBM not installed. Skipping advanced models.")
            print("   To install: pip install xgboost lightgbm")

# ===================== Quick Training Script =====================

def quick_train():
    """Quick training with minimal dependencies"""
    print("="*60)
    print("🤖 ENHANCED AI TRAINING SYSTEM")
    print("="*60)
    
    trainer = SimpleAITrainer()
    
    try:
        # Step 1: Load data
        df = trainer.load_all_data()
        
        # Step 2: Create features
        features_df = trainer.create_features(df)
        
        # Step 3: Train models
        scores = trainer.train_models(features_df)
        
        # Step 4: Save models
        trainer.save_models()
        
        print("\n🎉 Training completed successfully!")
        print("\n📊 Model Performance Summary:")
        for model, score in scores.items():
            print(f"  {model}: {score:.2%} accuracy")
        
        print("\n✅ Models are ready to use!")
        print("   You can now run: python main_enhanced.py")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("   Please check your data files in the 'data' folder")

# ===================== Full Training Script =====================

def full_train():
    """Full training with all features"""
    print("="*60)
    print("🤖 FULL AI TRAINING SYSTEM")
    print("="*60)
    
    trainer = AdvancedAITrainer()
    
    try:
        # Load and prepare data
        df = trainer.load_all_data()
        features_df = trainer.create_features(df)
        
        # Train basic models
        scores = trainer.train_models(features_df)
        
        # Train advanced models
        trainer.train_advanced_models(features_df)
        
        # Save everything
        trainer.save_models()
        
        # Save advanced models
        if trainer.advanced_models:
            model_dir = 'ai_models/simple_enhanced'
            for name, model in trainer.advanced_models.items():
                model_path = os.path.join(model_dir, f'{name}_model.pkl')
                joblib.dump(model, model_path)
                print(f"💾 Saved {name} to {model_path}")
        
        print("\n🎉 Full training completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

# ===================== Main Entry Point =====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Enhanced AI Models')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Training mode: quick (basic) or full (with advanced models)')
    parser.add_argument('--data-path', default='data',
                       help='Path to data folder containing CSV files')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_train()
    else:
        full_train()
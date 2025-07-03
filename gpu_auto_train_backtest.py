#!/usr/bin/env python3
# gpu_auto_train_backtest.py
# ===========================
# GPU-Accelerated Auto-Training & Backtesting System
# Supports CUDA acceleration for faster training

import os
import sys
import argparse
import json
import joblib
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
import time
import random
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated libraries
try:
    import cupy as cp  # GPU arrays
    import cudf  # GPU DataFrames
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.ensemble import RandomForestRegressor as cuRFR
    from cuml.svm import SVC as cuSVC
    from cuml.neural_network import MLPClassifier as cuMLP
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    from rapids_singlecell import preprocessing as rsc_pp
    GPU_AVAILABLE = True
    print("✅ GPU acceleration available (RAPIDS)")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPU acceleration not available, using CPU")

# PyTorch for neural networks with GPU
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import LabelEncoder
    
    TORCH_GPU_AVAILABLE = torch.cuda.is_available()
    if TORCH_GPU_AVAILABLE:
        print(f"✅ PyTorch GPU available: {torch.cuda.get_device_name(0)}")
        DEVICE = torch.device("cuda")
    else:
        print("⚠️ PyTorch GPU not available, using CPU")
        DEVICE = torch.device("cpu")
except ImportError:
    TORCH_GPU_AVAILABLE = False
    print("⚠️ PyTorch not installed")

# XGBoost with GPU support
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not installed")

# Import from existing modules
try:
    from train_enhanced_ai import SimpleAITrainer
except ImportError:
    class SimpleAITrainer:
        def __init__(self, data_path='data'):
            self.data_path = data_path
            
        def load_all_data(self):
            all_data = []
            for file in os.listdir(self.data_path):
                if file.endswith('_yfinance_daily.csv'):
                    symbol = file.replace('_yfinance_daily.csv', '')
                    file_path = os.path.join(self.data_path, file)
                    try:
                        df = pd.read_csv(file_path)
                        if 'Date' not in df.columns:
                            df = pd.read_csv(file_path, header=[0,1,2], index_col=0, parse_dates=True)
                            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                            df.reset_index(inplace=True)
                            df.rename(columns={'index': 'Date'}, inplace=True)
                        df['Symbol'] = symbol
                        df['Date'] = pd.to_datetime(df['Date'])
                        all_data.append(df)
                        print(f"✅ Loaded {symbol}: {len(df)} rows")
                    except Exception as e:
                        print(f"❌ Error loading {file}: {e}")
            
            if not all_data:
                raise ValueError("No data files found!")
            
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.sort_values(['Symbol', 'Date'], inplace=True)
            return combined_df

# ===================== GPU-Accelerated Neural Network =====================

class TradingNet(nn.Module):
    """PyTorch Neural Network for GPU acceleration"""
    
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout=0.3):
        super(TradingNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 3))  # 3 classes: SELL, HOLD, BUY
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class GPUNeuralNetworkClassifier:
    """Sklearn-compatible wrapper for PyTorch neural network"""
    
    def __init__(self, input_size=None, hidden_sizes=[256, 128, 64], 
                 learning_rate=0.001, epochs=100, batch_size=64):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def fit(self, X, y):
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Set input size
        if self.input_size is None:
            self.input_size = X.shape[1]
        
        # Create model
        self.model = TradingNet(self.input_size, self.hidden_sizes).to(DEVICE)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        y_tensor = torch.LongTensor(y_encoded).to(DEVICE)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if epoch % 20 == 0:
                print(f"    Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        return self
    
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
            
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X):
        if hasattr(X, 'values'):
            X = X.values
            
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            
        return probabilities

# ===================== Enhanced Configuration with GPU Options =====================

class GPUAutoTrainConfig:
    """Configuration for GPU-accelerated auto-training system"""
    TARGET_WIN_RATE = 80.0
    MAX_ITERATIONS = 100
    MIN_TRADES = 20
    
    # GPU settings
    USE_GPU = GPU_AVAILABLE or TORCH_GPU_AVAILABLE
    GPU_BATCH_SIZE = 1024  # Larger batch size for GPU
    
    # Model parameters for GPU models
    PARAM_GRIDS = {
        'RandomForest': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        },
        'NeuralNetwork': {
            'hidden_sizes': [[256, 128, 64], [512, 256, 128], [128, 64, 32]],
            'learning_rate': [0.001, 0.01, 0.1],
            'epochs': [50, 100, 150],
            'batch_size': [64, 128, 256]
        }
    }
    
    # Feature engineering options (same as before)
    FEATURE_OPTIONS = {
        'technical_periods': [5, 10, 20, 30, 50, 100, 200],
        'momentum_periods': [5, 10, 14, 20],
        'volatility_windows': [10, 20, 30],
        'volume_indicators': ['vwap', 'obv', 'ad', 'mfi'],
        'pattern_recognition': ['candlestick', 'chart_patterns']
    }

# ===================== GPU-Accelerated Feature Engineering =====================

class GPUFeatureEngineer:
    """GPU-accelerated feature engineering using CuPy/CuDF"""
    
    def __init__(self, config: GPUAutoTrainConfig):
        self.config = config
        self.use_gpu = config.USE_GPU and GPU_AVAILABLE
        
    def create_advanced_features(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Create features with GPU acceleration when available"""
        
        if self.use_gpu:
            # Convert to GPU DataFrame
            gdf = cudf.from_pandas(df)
            features = self._create_features_gpu(gdf, iteration)
            # Convert back to pandas
            return features.to_pandas()
        else:
            # Use CPU version
            return self._create_features_cpu(df, iteration)
    
    def _create_features_gpu(self, gdf, iteration):
        """GPU-accelerated feature creation using CuDF"""
        features = gdf.copy()
        
        # Sort by Symbol and Date
        features = features.sort_values(['Symbol', 'Date'])
        
        # Random seed for feature selection
        random.seed(iteration)
        
        # Price-based features (GPU-accelerated)
        for period in random.sample(self.config.FEATURE_OPTIONS['technical_periods'], 4):
            features[f'SMA_{period}'] = features.groupby('Symbol')['Close'].rolling(
                window=period, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            features[f'EMA_{period}'] = features.groupby('Symbol')['Close'].ewm(
                span=period, adjust=False
            ).mean().reset_index(0, drop=True)
            
            features[f'Close_to_SMA_{period}'] = features['Close'] / features[f'SMA_{period}']
        
        # RSI (GPU implementation)
        for period in random.sample(self.config.FEATURE_OPTIONS['momentum_periods'], 3):
            features[f'RSI_{period}'] = features.groupby('Symbol')['Close'].apply(
                lambda x: self._calculate_rsi_gpu(x, period)
            ).reset_index(0, drop=True)
        
        # Volume features
        features['Volume_MA20'] = features.groupby('Symbol')['Volume'].rolling(
            window=20, min_periods=1
        ).mean().reset_index(0, drop=True)
        features['Volume_Ratio'] = features['Volume'] / features['Volume_MA20']
        
        # MACD
        features['EMA_12'] = features.groupby('Symbol')['Close'].ewm(
            span=12, adjust=False
        ).mean().reset_index(0, drop=True)
        features['EMA_26'] = features.groupby('Symbol')['Close'].ewm(
            span=26, adjust=False
        ).mean().reset_index(0, drop=True)
        features['MACD'] = features['EMA_12'] - features['EMA_26']
        features['MACD_Signal'] = features.groupby('Symbol')['MACD'].ewm(
            span=9, adjust=False
        ).mean().reset_index(0, drop=True)
        
        return features
    
    def _create_features_cpu(self, df, iteration):
        """CPU version of feature creation (fallback)"""
        features = df.copy()
        features = features.sort_values(['Symbol', 'Date'])
        
        random.seed(iteration)
        
        # Price-based features
        for period in random.sample(self.config.FEATURE_OPTIONS['technical_periods'], 4):
            features[f'SMA_{period}'] = features.groupby('Symbol')['Close'].transform(
                lambda x: x.rolling(window=period, min_periods=1).mean()
            )
            features[f'EMA_{period}'] = features.groupby('Symbol')['Close'].transform(
                lambda x: x.ewm(span=period, adjust=False).mean()
            )
            features[f'Close_to_SMA_{period}'] = features['Close'] / features[f'SMA_{period}']
        
        # RSI
        for period in random.sample(self.config.FEATURE_OPTIONS['momentum_periods'], 3):
            features[f'RSI_{period}'] = features.groupby('Symbol')['Close'].transform(
                lambda x: self._calculate_rsi_cpu(x, period)
            )
        
        # Volume features
        features['Volume_MA20'] = features.groupby('Symbol')['Volume'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        )
        features['Volume_Ratio'] = features['Volume'] / features['Volume_MA20']
        
        # MACD
        features['EMA_12'] = features.groupby('Symbol')['Close'].transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )
        features['EMA_26'] = features.groupby('Symbol')['Close'].transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )
        features['MACD'] = features['EMA_12'] - features['EMA_26']
        features['MACD_Signal'] = features.groupby('Symbol')['MACD'].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
        
        # Additional features
        features['High_Low_Ratio'] = features['High'] / features['Low']
        features['Close_Open_Ratio'] = features['Close'] / features['Open']
        
        return features
    
    def _calculate_rsi_gpu(self, prices, period):
        """GPU-accelerated RSI calculation"""
        if GPU_AVAILABLE:
            # Convert to CuPy array
            prices_gpu = cp.asarray(prices.values)
            delta = cp.diff(prices_gpu)
            gain = cp.where(delta > 0, delta, 0)
            loss = cp.where(delta < 0, -delta, 0)
            
            # Calculate rolling means
            avg_gain = cp.convolve(gain, cp.ones(period)/period, mode='valid')
            avg_loss = cp.convolve(loss, cp.ones(period)/period, mode='valid')
            
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # Pad to match original length
            rsi_padded = cp.concatenate([cp.full(period, 50), rsi])
            
            return cudf.Series(rsi_padded[:len(prices)])
        else:
            return self._calculate_rsi_cpu(prices, period)
    
    def _calculate_rsi_cpu(self, prices, period):
        """CPU RSI calculation"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

# ===================== GPU-Accelerated AI Trainer =====================

class GPUAutoAITrainer(SimpleAITrainer):
    """GPU-accelerated AI trainer"""
    
    def __init__(self, data_path='data', config=None):
        super().__init__(data_path)
        self.config = config or GPUAutoTrainConfig()
        self.feature_engineer = GPUFeatureEngineer(self.config)
        self.training_history = []
        
    def train_iteration(self, iteration: int, df_all: pd.DataFrame):
        """Train models with GPU acceleration"""
        print(f"\n🔄 Training Iteration {iteration}", flush=True)
        
        # Create features
        features_df = self.feature_engineer.create_advanced_features(df_all, iteration)
        
        # Add target
        features_df['Future_Return'] = features_df.groupby('Symbol')['Close'].pct_change().shift(-1)
        features_df['Target'] = features_df['Future_Return'].apply(
            lambda x: 2 if x > 0.02 else (0 if x < -0.02 else 1)
        )
        
        features_df = features_df.dropna()
        
        # Prepare features
        feature_cols = [col for col in features_df.columns 
                       if col not in ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 
                                     'Volume', 'Future_Return', 'Target']]
        
        X = features_df[feature_cols]
        y = features_df['Target']
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        models = {}
        scores = {}
        
        # Choose models based on available libraries
        if self.config.USE_GPU:
            print("  🚀 Using GPU acceleration for training", flush=True)
            
            # Scale features
            if GPU_AVAILABLE:
                scaler = cuStandardScaler()
            else:
                scaler = StandardScaler()
            
            X_scaled = scaler.fit_transform(X)
            
            # GPU Random Forest (if RAPIDS available)
            if GPU_AVAILABLE:
                print("  Training GPU Random Forest...", flush=True)
                rf = cuRF(n_estimators=200, max_depth=20, random_state=iteration)
                rf.fit(X_scaled, y)
                models['RandomForest'] = rf
                scores['RandomForest'] = 0.0  # cuML doesn't have built-in CV
            
            # XGBoost with GPU
            if XGB_AVAILABLE:
                print("  Training XGBoost (GPU)...", flush=True)
                xgb_params = {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 7,
                    'tree_method': 'gpu_hist',  # GPU acceleration
                    'predictor': 'gpu_predictor',
                    'random_state': iteration
                }
                xgb_model = xgb.XGBClassifier(**xgb_params)
                
                # Simple CV for XGBoost
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    xgb_model.fit(X_train, y_train)
                    score = xgb_model.score(X_val, y_val)
                    cv_scores.append(score)
                
                xgb_model.fit(X_scaled, y)  # Final fit on all data
                models['XGBoost'] = xgb_model
                scores['XGBoost'] = np.mean(cv_scores)
            
            # PyTorch Neural Network
            if TORCH_GPU_AVAILABLE:
                print("  Training Neural Network (GPU)...", flush=True)
                nn_model = GPUNeuralNetworkClassifier(
                    input_size=X_scaled.shape[1],
                    hidden_sizes=[256, 128, 64],
                    learning_rate=0.001,
                    epochs=100,
                    batch_size=256
                )
                nn_model.fit(X_scaled, y)
                models['NeuralNetwork'] = nn_model
                scores['NeuralNetwork'] = 0.0  # Will evaluate separately
        
        else:
            # Fallback to CPU models
            print("  Using CPU for training", flush=True)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Random Forest
            print("  Training Random Forest (CPU)...", flush=True)
            rf = RandomForestClassifier(n_estimators=100, random_state=iteration, n_jobs=-1)
            rf.fit(X_scaled, y)
            models['RandomForest'] = rf
            
            # Gradient Boosting
            print("  Training Gradient Boosting (CPU)...", flush=True)
            gb = GradientBoostingClassifier(n_estimators=100, random_state=iteration)
            gb.fit(X_scaled, y)
            models['GradientBoosting'] = gb
        
        # Create ensemble
        if len(models) > 1:
            ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in models.items()],
                voting='soft'
            )
            ensemble.fit(X_scaled, y)
        else:
            ensemble = list(models.values())[0]
        
        # Save models
        model_dir = f'ai_models/iteration_{iteration}'
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
        
        with open(os.path.join(model_dir, 'features.json'), 'w') as f:
            json.dump(feature_cols, f)
        
        for name, model in models.items():
            # Handle GPU models specially
            if GPU_AVAILABLE and isinstance(model, (cuRF, cuSVC)):
                # Save as pickle for GPU models
                import pickle
                with open(os.path.join(model_dir, f'{name}_model.pkl'), 'wb') as f:
                    pickle.dump(model, f)
            else:
                joblib.dump(model, os.path.join(model_dir, f'{name}_model.pkl'))
        
        joblib.dump(ensemble, os.path.join(model_dir, 'ensemble_model.pkl'))
        
        # Store training info
        training_info = {
            'iteration': iteration,
            'feature_count': len(feature_cols),
            'training_samples': len(X),
            'model_scores': scores,
            'gpu_used': self.config.USE_GPU
        }
        
        self.training_history.append(training_info)
        
        print(f"  ✅ Training completed. Scores: {scores}", flush=True)
        
        return model_dir

# ===================== Rest of the system remains similar =====================

# Copy the rest of the classes from the original file:
# - AutoAIBacktestStrategy (no changes needed)
# - AutoTrainingSystem (use GPUAutoAITrainer instead)
# - Main function (add GPU info display)

class AutoAIBacktestStrategy(bt.Strategy):
    """Backtest strategy (same as before, works with GPU-trained models)"""
    
    params = dict(
        model_dir='ai_models/iteration_0',
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        position_size_pct=0.95,
        use_ensemble=True
    )
    
    def __init__(self):
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        self.trades = []
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Load models
        model_dir = self.p.model_dir
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory '{model_dir}' not found.")
        
        # Load scaler and features
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        with open(os.path.join(model_dir, 'features.json')) as f:
            self.feature_names = json.load(f)
        
        # Load models
        if self.p.use_ensemble and os.path.exists(os.path.join(model_dir, 'ensemble_model.pkl')):
            self.model = joblib.load(os.path.join(model_dir, 'ensemble_model.pkl'))
            print(f"🤖 Loaded ensemble model", flush=True)
        else:
            model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl') and f != 'ensemble_model.pkl']
            self.models = []
            for m in model_files:
                try:
                    model = joblib.load(os.path.join(model_dir, m))
                    self.models.append(model)
                except:
                    # Try pickle for GPU models
                    import pickle
                    with open(os.path.join(model_dir, m), 'rb') as f:
                        model = pickle.load(f)
                        self.models.append(model)
            
            print(f"🤖 Loaded {len(self.models)} individual models", flush=True)
    
    def next(self):
        # Same implementation as before
        pass
    
    def notify_order(self, order):
        # Same implementation as before
        pass
    
    def stop(self):
        # Same implementation as before
        pass

class GPUAutoTrainingSystem:
    """GPU-accelerated auto-training system"""
    
    def __init__(self, config=None):
        self.config = config or GPUAutoTrainConfig()
        self.trainer = GPUAutoAITrainer(config=self.config)
        self.results_history = []
        
        # Display GPU info
        print("\n🖥️ System Information:")
        print(f"  GPU Available: {self.config.USE_GPU}")
        if TORCH_GPU_AVAILABLE:
            print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        if GPU_AVAILABLE:
            print(f"  RAPIDS/cuML: Available")
        if XGB_AVAILABLE:
            print(f"  XGBoost: Available")
    
    # Rest of the methods remain the same as AutoTrainingSystem
    def run_backtest(self, model_dir: str, symbols: list, start_cash: float = 10000.0):
        # Same as before
        pass
    
    def evaluate_results(self, results: dict):
        # Same as before
        pass
    
    def auto_train_loop(self, symbols: list, max_iterations: int = None):
        # Same as before, but uses GPU trainer
        pass

# ===================== Main Entry Point =====================

def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated Auto-Training and Backtesting System")
    parser.add_argument('--target-win-rate', type=float, default=80.0, help='Target win rate percentage')
    parser.add_argument('--max-iterations', type=int, default=100, help='Maximum training iterations')
    parser.add_argument('--symbols', nargs='+', default=['BTC', 'ETH', 'LTC', 'BNB'], help='Symbols to trade')
    parser.add_argument('--data-path', default='data', help='Path to data folder')
    parser.add_argument('--cash', type=float, default=10000.0, help='Starting cash for backtest')
    parser.add_argument('--plot', action='store_true', help='Plot training progress')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # Configure system
    config = GPUAutoTrainConfig()
    config.TARGET_WIN_RATE = args.target_win_rate
    config.MAX_ITERATIONS = args.max_iterations
    
    if args.no_gpu:
        config.USE_GPU = False
        print("ℹ️ GPU acceleration disabled by user")
    
    # Run auto-training system
    system = GPUAutoTrainingSystem(config)
    best_model_dir, best_win_rate = system.auto_train_loop(args.symbols, args.max_iterations)
    
    print(f"\n✅ Auto-training completed!", flush=True)
    print(f"   Best Model: {best_model_dir}", flush=True)
    print(f"   Best Win Rate: {best_win_rate:.1f}%", flush=True)

if __name__ == '__main__':
    main()
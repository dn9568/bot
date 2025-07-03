# advanced_trading_bot.py - Enhanced AI Trading System
# ===============================================================
import os
import sys
import time
import json
import pickle
import logging
import asyncio
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pandas as pd
import joblib

# Machine Learning & Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from cuml.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    IsolationForest
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb

# Technical Analysis (without talib)
from scipy import signal
from scipy.stats import linregress
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Binance API
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException

# Async & Performance
import aiohttp
import asyncio
from functools import lru_cache
import numba as nb

warnings.filterwarnings('ignore')

# ===================== Configuration =====================
@dataclass
class TradingConfig:
    """Enhanced configuration with dynamic parameters"""
    # API Settings
    binance_api_key: str = "YOUR_API_KEY"
    binance_api_secret: str = "YOUR_API_SECRET"
    
    # Trading Parameters
    initial_balance: float = 10000.0
    base_trade_amount: float = 100.0
    max_positions: int = 3
    leverage: int = 10
    
    # Risk Management
    max_drawdown: float = 0.15
    position_sizing_method: str = "kelly"  # kelly, fixed, volatility_adjusted
    risk_per_trade: float = 0.02
    
    # AI Model Settings
    ensemble_models: List[str] = field(default_factory=lambda: [
        'transformer', 'lstm', 'gru', 'xgboost', 'lightgbm', 'catboost'
    ])
    confidence_threshold: float = 0.65
    min_data_points: int = 300
    
    # Market Analysis
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '1h'])
    indicators_config: Dict = field(default_factory=lambda: {
        'ema': [9, 21, 50, 100, 200],
        'rsi': [14, 21],
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'bb': {'period': 20, 'std': 2},
        'atr': [14, 21],
        'vwap': True,
        'market_profile': True
    })
    
    # Execution
    update_interval: int = 30  # seconds
    use_websocket: bool = True
    parallel_processing: bool = True
    
    # Advanced Features
    use_sentiment_analysis: bool = True
    use_on_chain_metrics: bool = True
    use_order_flow: bool = True
    
    # Paths
    model_path: str = "ai_models/advanced_v2"
    log_path: str = "logs/trading_v2.log"
    data_path: str = "data/market_data.pkl"

# ===================== Deep Learning Models =====================

class TransformerPredictor(nn.Module):
    """Transformer-based price prediction model"""
    def __init__(self, input_dim=50, d_model=128, nhead=8, num_layers=4, output_dim=3):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc_layers(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class LSTMGRUHybrid(nn.Module):
    """Hybrid LSTM-GRU model for time series prediction"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2, bidirectional=True)
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, num_layers, 
                         batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(lstm_out)
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        return self.fc(attn_out[:, -1, :])

# ===================== Advanced Technical Indicators =====================

class AdvancedIndicators:
    """Custom technical indicators without talib dependency"""
    
    @staticmethod
    @nb.jit(nopython=True)
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    @staticmethod
    @nb.jit(nopython=True)
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        deltas = np.diff(data)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(data)
        rsi[:period] = np.nan
        rsi[period] = 100 - (100 / (1 + rs))
        
        for i in range(period + 1, len(data)):
            delta = deltas[i-1]
            if delta > 0:
                up = (up * (period - 1) + delta) / period
                down = (down * (period - 1)) / period
            else:
                up = (up * (period - 1)) / period
                down = (down * (period - 1) - delta) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD indicator"""
        ema_fast = AdvancedIndicators.ema(data, fast)
        ema_slow = AdvancedIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = AdvancedIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2.0):
        """Bollinger Bands"""
        sma = pd.Series(data).rolling(period).mean().values
        std = pd.Series(data).rolling(period).std().values
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray):
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return np.cumsum(typical_price * volume) / np.cumsum(volume)
    
    @staticmethod
    def market_profile(prices: np.ndarray, volumes: np.ndarray, bins: int = 50):
        """Market Profile - Volume at Price"""
        price_range = np.linspace(prices.min(), prices.max(), bins)
        profile = np.zeros(bins - 1)
        
        for i in range(len(price_range) - 1):
            mask = (prices >= price_range[i]) & (prices < price_range[i + 1])
            profile[i] = volumes[mask].sum()
        
        poc_idx = np.argmax(profile)  # Point of Control
        value_area = np.percentile(profile, [30, 70])
        
        return {
            'profile': profile,
            'poc': price_range[poc_idx],
            'value_area_low': price_range[np.where(profile >= value_area[0])[0][0]],
            'value_area_high': price_range[np.where(profile >= value_area[1])[0][-1]]
        }

# ===================== Feature Engineering =====================

class FeatureEngineer:
    """Advanced feature engineering for ML models"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.indicators = AdvancedIndicators()
        self.scaler = RobustScaler()
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        features = df.copy()
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['price_range'] = (df['high'] - df['low']) / df['close']
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        # Technical indicators
        for period in self.config.indicators_config['ema']:
            features[f'ema_{period}'] = self.indicators.ema(df['close'].values, period)
            features[f'ema_ratio_{period}'] = df['close'] / features[f'ema_{period}']
        
        for period in self.config.indicators_config['rsi']:
            features[f'rsi_{period}'] = self.indicators.rsi(df['close'].values, period)
        
        # MACD
        macd_config = self.config.indicators_config['macd']
        macd_line, signal_line, histogram = self.indicators.macd(
            df['close'].values, 
            macd_config['fast'], 
            macd_config['slow'], 
            macd_config['signal']
        )
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_hist'] = histogram
        
        # Bollinger Bands
        bb_config = self.config.indicators_config['bb']
        upper, middle, lower = self.indicators.bollinger_bands(
            df['close'].values, 
            bb_config['period'], 
            bb_config['std']
        )
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = (upper - lower) / middle
        features['bb_position'] = (df['close'] - lower) / (upper - lower)
        
        # Market microstructure
        features['bid_ask_spread'] = df['ask'] - df['bid'] if 'bid' in df else 0
        features['order_flow_imbalance'] = self._calculate_order_flow(df)
        
        # Statistical features
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = df['returns'].rolling(window).std()
            features[f'skew_{window}'] = df['returns'].rolling(window).skew()
            features[f'kurtosis_{window}'] = df['returns'].rolling(window).kurt()
        
        # Regime detection
        features['regime'] = self._detect_market_regime(df)
        
        # Fourier features for cyclical patterns
        features.update(self._fourier_features(df['close'].values))
        
        # Lag features
        for lag in [1, 5, 10, 20]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_lag_{lag}'] = features['volume_ratio'].shift(lag)
        
        return features.dropna()
    
    def _calculate_order_flow(self, df: pd.DataFrame) -> pd.Series:
        """Calculate order flow imbalance"""
        if 'bid_volume' in df and 'ask_volume' in df:
            return (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
        else:
            # Approximate using price and volume
            price_dir = np.sign(df['close'].diff())
            return price_dir * df['volume']
    
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect market regime using Hidden Markov Model"""
        returns = df['close'].pct_change().dropna()
        
        # Simple regime detection based on volatility and trend
        vol = returns.rolling(20).std()
        trend = df['close'].rolling(50).mean()
        
        regime = pd.Series(index=df.index, dtype=int)
        regime[(vol < vol.quantile(0.3)) & (df['close'] > trend)] = 0  # Low vol uptrend
        regime[(vol < vol.quantile(0.3)) & (df['close'] <= trend)] = 1  # Low vol downtrend
        regime[(vol >= vol.quantile(0.7)) & (df['close'] > trend)] = 2  # High vol uptrend
        regime[(vol >= vol.quantile(0.7)) & (df['close'] <= trend)] = 3  # High vol downtrend
        regime.fillna(method='ffill', inplace=True)
        
        return regime
    
    def _fourier_features(self, prices: np.ndarray, n_features: int = 10) -> Dict[str, np.ndarray]:
        """Extract Fourier features for cyclical patterns"""
        fft = np.fft.fft(prices)
        frequencies = np.fft.fftfreq(len(prices))
        
        # Get top frequency components
        magnitudes = np.abs(fft)
        top_indices = np.argsort(magnitudes)[-n_features:]
        
        features = {}
        for i, idx in enumerate(top_indices):
            features[f'fourier_mag_{i}'] = magnitudes[idx]
            features[f'fourier_freq_{i}'] = frequencies[idx]
        
        return features

# ===================== AI Trading Engine =====================

class AITradingEngine:
    """Main AI trading engine with ensemble models"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.models = {}
        self.feature_engineer = FeatureEngineer(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self._initialize_models()
        
        # Market data storage
        self.market_data = defaultdict(lambda: deque(maxlen=5000))
        self.predictions_history = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
        
    def _initialize_models(self):
        """Initialize all AI models"""
        # Deep Learning models
        self.models['transformer'] = TransformerPredictor(
            input_dim=100, d_model=128, nhead=8, num_layers=4
        ).to(self.device)
        
        self.models['lstm_gru'] = LSTMGRUHybrid(
            input_dim=100, hidden_dim=128, num_layers=3
        ).to(self.device)
        
        # Gradient Boosting models
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.01,
            objective='multi:softprob',
            use_label_encoder=False
        )
        
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.01,
            objective='multiclass',
            num_class=3
        )
        
        # Anomaly detection
        self.models['anomaly'] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
    def update_market_data(self, symbol: str, data: Dict[str, float]):
        """Update market data for a symbol"""
        self.market_data[symbol].append({
            'timestamp': datetime.now(timezone.utc),
            **data
        })
    
    def predict(self, symbol: str) -> Tuple[int, float, str]:
        """
        Make prediction for a symbol
        Returns: (action, confidence, reasoning)
        action: -1 (sell), 0 (hold), 1 (buy)
        """
        # Get market data
        data = list(self.market_data[symbol])
        if len(data) < self.config.min_data_points:
            return 0, 0.0, "Insufficient data"
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Engineer features
        features_df = self.feature_engineer.create_features(df)
        if len(features_df) < 50:
            return 0, 0.0, "Insufficient features"
        
        # Get latest features
        X = features_df.iloc[-1:].values
        
        # Get predictions from all models
        predictions = []
        confidences = []
        
        # Deep Learning predictions
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Transformer
            trans_pred = F.softmax(self.models['transformer'](X_tensor.unsqueeze(0)), dim=1)
            predictions.append(trans_pred.argmax().item() - 1)  # Convert to -1, 0, 1
            confidences.append(trans_pred.max().item())
            
            # LSTM-GRU
            lstm_pred = F.softmax(self.models['lstm_gru'](X_tensor.unsqueeze(0)), dim=1)
            predictions.append(lstm_pred.argmax().item() - 1)
            confidences.append(lstm_pred.max().item())
        
        # ML predictions (if models are trained)
        if hasattr(self.models['xgboost'], 'classes_'):
            xgb_pred = self.models['xgboost'].predict_proba(X)[0]
            predictions.append(np.argmax(xgb_pred) - 1)
            confidences.append(np.max(xgb_pred))
            
            lgb_pred = self.models['lightgbm'].predict_proba(X)[0]
            predictions.append(np.argmax(lgb_pred) - 1)
            confidences.append(np.max(lgb_pred))
        
        # Ensemble decision
        action = int(np.sign(np.mean(predictions)))
        confidence = np.mean(confidences)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(features_df.iloc[-1], predictions, confidences)
        
        # Adjust for market regime
        regime = features_df.iloc[-1]['regime']
        if regime in [1, 3] and confidence < 0.8:  # Downtrend or high vol downtrend
            action = min(action, 0)  # Don't buy in downtrend unless very confident
        
        # Record prediction
        self.predictions_history.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning
        })
        
        return action, confidence, reasoning
    
    def _generate_reasoning(self, features: pd.Series, predictions: List[int], confidences: List[float]) -> str:
        """Generate human-readable reasoning for the prediction"""
        reasons = []
        
        # Technical indicators
        if features['rsi_14'] < 30:
            reasons.append(f"RSI oversold ({features['rsi_14']:.1f})")
        elif features['rsi_14'] > 70:
            reasons.append(f"RSI overbought ({features['rsi_14']:.1f})")
        
        if features['macd_hist'] > 0:
            reasons.append("MACD bullish crossover")
        elif features['macd_hist'] < 0:
            reasons.append("MACD bearish crossover")
        
        # Market regime
        regime_names = ['Low Vol Uptrend', 'Low Vol Downtrend', 'High Vol Uptrend', 'High Vol Downtrend']
        reasons.append(f"Market regime: {regime_names[int(features['regime'])]}")
        
        # Model consensus
        consensus = sum(predictions) / len(predictions)
        if consensus > 0.5:
            reasons.append(f"Strong buy consensus ({consensus:.1f})")
        elif consensus < -0.5:
            reasons.append(f"Strong sell consensus ({consensus:.1f})")
        
        # Confidence
        avg_confidence = np.mean(confidences)
        reasons.append(f"Model confidence: {avg_confidence:.1%}")
        
        return " | ".join(reasons[:4])
    
    def train_models(self, historical_data: pd.DataFrame):
        """Train all models on historical data"""
        print("🚀 Training AI models...")
        
        # Engineer features
        features_df = self.feature_engineer.create_features(historical_data)
        
        # Create labels (future returns)
        features_df['future_return'] = features_df['close'].pct_change().shift(-1)
        features_df['label'] = pd.cut(
            features_df['future_return'],
            bins=[-np.inf, -0.001, 0.001, np.inf],
            labels=[0, 1, 2]  # sell, hold, buy
        )
        features_df.dropna(inplace=True)
        
        # Prepare data
        feature_cols = [col for col in features_df.columns if col not in ['label', 'future_return']]
        X = features_df[feature_cols].values
        y = features_df['label'].values
        
        # Scale features
        X_scaled = self.feature_engineer.scaler.fit_transform(X)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train ML models
            self.models['xgboost'].fit(X_train, y_train)
            self.models['lightgbm'].fit(X_train, y_train)
            
            # Train deep learning models
            self._train_deep_models(X_train, y_train, X_val, y_val)
        
        print("✅ Model training completed!")
    
    def _train_deep_models(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train deep learning models"""
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Train each deep model
        for model_name in ['transformer', 'lstm_gru']:
            model = self.models[model_name]
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch.unsqueeze(1))  # Add sequence dimension
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                correct = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        outputs = model(X_batch.unsqueeze(1))
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                        
                        _, predicted = outputs.max(1)
                        correct += predicted.eq(y_batch).sum().item()
                
                if epoch % 10 == 0:
                    print(f"{model_name} - Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Acc: {correct/len(val_dataset):.4f}")

# ===================== Risk Management =====================

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.positions = {}
        self.balance = config.initial_balance
        self.equity_curve = deque(maxlen=1000)
        self.drawdown_history = deque(maxlen=1000)
        
    def calculate_position_size(self, symbol: str, confidence: float, volatility: float) -> float:
        """Calculate optimal position size"""
        if self.config.position_sizing_method == 'kelly':
            return self._kelly_criterion(confidence, volatility)
        elif self.config.position_sizing_method == 'volatility_adjusted':
            return self._volatility_adjusted_size(volatility)
        else:
            return self.config.base_trade_amount
    
    def _kelly_criterion(self, win_prob: float, avg_win_loss_ratio: float = 1.5) -> float:
        """Kelly Criterion for position sizing"""
        if win_prob <= 0 or win_prob >= 1:
            return 0
        
        # Kelly percentage
        kelly_pct = (win_prob * avg_win_loss_ratio - (1 - win_prob)) / avg_win_loss_ratio
        
        # Apply Kelly fraction (usually 0.25 for safety)
        kelly_pct = max(0, min(kelly_pct * 0.25, 0.1))
        
        return self.balance * kelly_pct
    
    def _volatility_adjusted_size(self, volatility: float) -> float:
        """Adjust position size based on volatility"""
        base_volatility = 0.02  # 2% baseline volatility
        adjustment_factor = base_volatility / max(volatility, 0.001)
        
        return self.config.base_trade_amount * adjustment_factor
    
    def check_risk_limits(self, symbol: str, action: int, size: float) -> bool:
        """Check if trade meets risk requirements"""
        # Check drawdown limit
        current_drawdown = self._calculate_drawdown()
        if current_drawdown > self.config.max_drawdown:
            return False
        
        # Check position limits
        if len(self.positions) >= self.config.max_positions and symbol not in self.positions:
            return False
        
        # Check exposure limit
        total_exposure = sum(pos['size'] * pos['current_price'] for pos in self.positions.values())
        if (total_exposure + size) / self.balance > 0.8:  # Max 80% exposure
            return False
        
        return True
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        if not self.equity_curve:
            return 0
        
        peak = max(self.equity_curve)
        current = self.equity_curve[-1]
        return (peak - current) / peak if peak > 0 else 0
    
    def update_position(self, symbol: str, action: int, price: float, size: float):
        """Update position after trade"""
        if action == 1:  # Buy
            if symbol in self.positions:
                # Average up
                pos = self.positions[symbol]
                total_size = pos['size'] + size
                pos['avg_price'] = (pos['avg_price'] * pos['size'] + price * size) / total_size
                pos['size'] = total_size
            else:
                self.positions[symbol] = {
                    'size': size,
                    'avg_price': price,
                    'entry_time': datetime.now(timezone.utc)
                }
            self.balance -= size * price
            
        elif action == -1 and symbol in self.positions:  # Sell
            pos = self.positions[symbol]
            pnl = (price - pos['avg_price']) * pos['size']
            self.balance += pos['size'] * price
            del self.positions[symbol]
            
            # Update performance metrics
            self._update_performance(pnl)
    
    def _update_performance(self, pnl: float):
        """Update performance tracking"""
        self.performance_metrics['total_trades'] += 1
        if pnl > 0:
            self.performance_metrics['winning_trades'] += 1
        self.performance_metrics['total_pnl'] += pnl
        
        # Update equity curve
        current_equity = self.balance + sum(
            pos['size'] * pos.get('current_price', pos['avg_price']) 
            for pos in self.positions.values()
        )
        self.equity_curve.append(current_equity)
        
        # Calculate Sharpe ratio
        if len(self.equity_curve) > 30:
            returns = np.diff(list(self.equity_curve))
            self.performance_metrics['sharpe_ratio'] = (
                np.mean(returns) / np.std(returns) * np.sqrt(252) 
                if np.std(returns) > 0 else 0
            )

# ===================== Market Data Manager =====================

class MarketDataManager:
    """Async market data management with WebSocket support"""
    
    def __init__(self, config: TradingConfig, client: Client):
        self.config = config
        self.client = client
        self.websocket_manager = None
        self.order_book = defaultdict(dict)
        self.trade_flow = defaultdict(deque)
        
    async def start_websocket_streams(self, symbols: List[str]):
        """Start WebSocket streams for real-time data"""
        if not self.config.use_websocket:
            return
        
        # Implementation depends on Binance WebSocket API
        # This is a placeholder for the actual WebSocket implementation
        pass
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data for a symbol"""
        try:
            # Get multiple timeframe data in parallel
            tasks = []
            for timeframe in self.config.timeframes:
                tasks.append(self._fetch_klines(symbol, timeframe))
            
            klines_data = await asyncio.gather(*tasks)
            
            # Get order book
            order_book = await self._fetch_order_book(symbol)
            
            # Get recent trades
            recent_trades = await self._fetch_recent_trades(symbol)
            
            # Aggregate data
            market_data = {
                'klines': dict(zip(self.config.timeframes, klines_data)),
                'order_book': order_book,
                'recent_trades': recent_trades,
                'timestamp': datetime.now(timezone.utc)
            }
            
            return market_data
            
        except Exception as e:
            logging.error(f"Error fetching market data for {symbol}: {e}")
            return {}
    
    async def _fetch_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch kline/candlestick data"""
        # Using sync client for now, can be replaced with async client
        klines = self.client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=500
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    async def _fetch_order_book(self, symbol: str, depth: int = 20) -> Dict:
        """Fetch order book data"""
        order_book = self.client.get_order_book(symbol=symbol, limit=depth)
        
        return {
            'bids': order_book['bids'],
            'asks': order_book['asks'],
            'timestamp': datetime.now(timezone.utc)
        }
    
    async def _fetch_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Fetch recent trades"""
        trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
        
        return [{
            'price': float(t['price']),
            'qty': float(t['qty']),
            'time': pd.to_datetime(t['time'], unit='ms'),
            'is_buyer_maker': t['isBuyerMaker']
        } for t in trades]

# ===================== Trading Executor =====================

class TradingExecutor:
    """Execute trades with advanced order management"""
    
    def __init__(self, config: TradingConfig, client: Client, risk_manager: RiskManager):
        self.config = config
        self.client = client
        self.risk_manager = risk_manager
        self.active_orders = {}
        self.execution_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'slippage': []})
        
    async def execute_trade(self, symbol: str, action: int, confidence: float, market_data: Dict):
        """Execute trade with smart order routing"""
        if action == 0:  # Hold
            return None
        
        # Get current market conditions
        current_price = float(market_data['klines']['1m']['close'].iloc[-1])
        volatility = market_data['klines']['1m']['close'].pct_change().std()
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(symbol, confidence, volatility)
        
        # Check risk limits
        if not self.risk_manager.check_risk_limits(symbol, action, position_size):
            logging.warning(f"Risk limits exceeded for {symbol}")
            return None
        
        # Determine order type and price
        order_type, order_price = self._determine_order_type(
            action, current_price, market_data['order_book'], volatility
        )
        
        # Execute order
        try:
            if self.config.paper_trading:
                # Simulate order
                order = self._simulate_order(symbol, action, position_size, order_price)
            else:
                # Real order
                order = await self._place_real_order(symbol, action, position_size, order_type, order_price)
            
            # Update risk manager
            self.risk_manager.update_position(symbol, action, order_price, position_size)
            
            # Track execution stats
            self._update_execution_stats(symbol, current_price, order_price)
            
            logging.info(f"Order executed: {symbol} {action} {position_size} @ {order_price}")
            return order
            
        except Exception as e:
            logging.error(f"Order execution failed: {e}")
            return None
    
    def _determine_order_type(self, action: int, current_price: float, 
                            order_book: Dict, volatility: float) -> Tuple[str, float]:
        """Determine optimal order type and price"""
        spread = float(order_book['asks'][0][0]) - float(order_book['bids'][0][0])
        spread_pct = spread / current_price
        
        # Use market order for high volatility or tight spreads
        if volatility > 0.02 or spread_pct < 0.0001:
            return 'MARKET', current_price
        
        # Use limit order with intelligent pricing
        if action == 1:  # Buy
            # Place order slightly above best bid
            limit_price = float(order_book['bids'][0][0]) * 1.0001
        else:  # Sell
            # Place order slightly below best ask
            limit_price = float(order_book['asks'][0][0]) * 0.9999
        
        return 'LIMIT', limit_price
    
    def _simulate_order(self, symbol: str, action: int, size: float, price: float) -> Dict:
        """Simulate order for paper trading"""
        return {
            'symbol': symbol,
            'side': 'BUY' if action == 1 else 'SELL',
            'size': size,
            'price': price,
            'status': 'FILLED',
            'timestamp': datetime.now(timezone.utc)
        }
    
    async def _place_real_order(self, symbol: str, action: int, size: float, 
                              order_type: str, price: float) -> Dict:
        """Place real order on exchange"""
        # Implementation for real order placement
        # This would use the Binance client to place actual orders
        pass
    
    def _update_execution_stats(self, symbol: str, expected_price: float, actual_price: float):
        """Track execution statistics"""
        slippage = abs(actual_price - expected_price) / expected_price
        
        self.execution_stats[symbol]['total'] += 1
        self.execution_stats[symbol]['success'] += 1
        self.execution_stats[symbol]['slippage'].append(slippage)

# ===================== Sentiment Analyzer =====================

class SentimentAnalyzer:
    """Analyze market sentiment from various sources"""
    
    def __init__(self):
        self.sentiment_scores = defaultdict(float)
        
    async def get_sentiment(self, symbol: str) -> float:
        """Get aggregate sentiment score for a symbol"""
        # Placeholder for sentiment analysis
        # Would integrate with news APIs, social media, etc.
        return 0.0

# ===================== Main Trading Bot =====================

class AdvancedTradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.client = Client(config.binance_api_key, config.binance_api_secret)
        
        # Initialize components
        self.ai_engine = AITradingEngine(config)
        self.risk_manager = RiskManager(config)
        self.market_data_manager = MarketDataManager(config, self.client)
        self.executor = TradingExecutor(config, self.client, self.risk_manager)
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Performance tracking
        self.start_time = datetime.now(timezone.utc)
        self.trade_history = []
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_path),
                logging.StreamHandler()
            ]
        )
    
    async def run(self):
        """Main trading loop"""
        logging.info("🚀 Starting Advanced AI Trading Bot")
        
        # Load historical data and train models if needed
        await self._initialize()
        
        while True:
            try:
                await self._trading_iteration()
                await asyncio.sleep(self.config.update_interval)
                
            except KeyboardInterrupt:
                logging.info("🛑 Shutting down bot...")
                break
            except Exception as e:
                logging.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _initialize(self):
        """Initialize bot with historical data"""
        logging.info("📊 Loading historical data...")
        
        for symbol in self.config.symbols:
            # Load historical data
            historical_df = await self.market_data_manager._fetch_klines(symbol, '1d')
            
            # Train models if needed
            if not os.path.exists(f"{self.config.model_path}/model_weights.pth"):
                self.ai_engine.train_models(historical_df)
                self._save_models()
            else:
                self._load_models()
        
        logging.info("✅ Initialization complete!")
    
    async def _trading_iteration(self):
        """Single iteration of the trading loop"""
        tasks = []
        
        # Fetch market data for all symbols in parallel
        for symbol in self.config.symbols:
            tasks.append(self._process_symbol(symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log performance
        self._log_performance()
    
    async def _process_symbol(self, symbol: str):
        """Process a single symbol"""
        try:
            # Get market data
            market_data = await self.market_data_manager.get_market_data(symbol)
            
            # Update AI engine with latest data
            latest_candle = market_data['klines']['1m'].iloc[-1]
            self.ai_engine.update_market_data(symbol, {
                'open': latest_candle['open'],
                'high': latest_candle['high'],
                'low': latest_candle['low'],
                'close': latest_candle['close'],
                'volume': latest_candle['volume']
            })
            
            # Get AI prediction
            action, confidence, reasoning = self.ai_engine.predict(symbol)
            
            # Get sentiment if enabled
            if self.config.use_sentiment_analysis:
                sentiment = await self.sentiment_analyzer.get_sentiment(symbol)
                # Adjust confidence based on sentiment
                confidence = confidence * 0.8 + sentiment * 0.2
            
            # Log prediction
            logging.info(f"{symbol}: Action={action}, Confidence={confidence:.2%}, Reason={reasoning}")
            
            # Execute trade if confident enough
            if confidence >= self.config.confidence_threshold:
                order = await self.executor.execute_trade(symbol, action, confidence, market_data)
                if order:
                    self.trade_history.append({
                        'timestamp': datetime.now(timezone.utc),
                        'symbol': symbol,
                        'action': action,
                        'confidence': confidence,
                        'reasoning': reasoning,
                        'order': order
                    })
            
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
    
    def _log_performance(self):
        """Log current performance metrics"""
        metrics = self.risk_manager.performance_metrics
        current_balance = self.risk_manager.balance
        
        # Calculate additional metrics
        runtime = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
        trades_per_hour = metrics['total_trades'] / runtime if runtime > 0 else 0
        win_rate = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
        
        logging.info(
            f"📊 Performance - "
            f"Balance: ${current_balance:.2f} | "
            f"PnL: ${metrics['total_pnl']:.2f} | "
            f"Trades: {metrics['total_trades']} | "
            f"Win Rate: {win_rate:.1%} | "
            f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
            f"Max DD: {metrics['max_drawdown']:.1%}"
        )
    
    def _save_models(self):
        """Save trained models"""
        os.makedirs(self.config.model_path, exist_ok=True)
        
        # Save PyTorch models
        for name in ['transformer', 'lstm_gru']:
            torch.save(
                self.ai_engine.models[name].state_dict(),
                f"{self.config.model_path}/{name}_weights.pth"
            )
        
        # Save sklearn models
        for name in ['xgboost', 'lightgbm']:
            joblib.dump(
                self.ai_engine.models[name],
                f"{self.config.model_path}/{name}_model.pkl"
            )
        
        # Save scaler
        joblib.dump(
            self.ai_engine.feature_engineer.scaler,
            f"{self.config.model_path}/scaler.pkl"
        )
        
        logging.info("💾 Models saved successfully!")
    
    def _load_models(self):
        """Load saved models"""
        # Load PyTorch models
        for name in ['transformer', 'lstm_gru']:
            self.ai_engine.models[name].load_state_dict(
                torch.load(f"{self.config.model_path}/{name}_weights.pth")
            )
        
        # Load sklearn models
        for name in ['xgboost', 'lightgbm']:
            self.ai_engine.models[name] = joblib.load(
                f"{self.config.model_path}/{name}_model.pkl"
            )
        
        # Load scaler
        self.ai_engine.feature_engineer.scaler = joblib.load(
            f"{self.config.model_path}/scaler.pkl"
        )
        
        logging.info("📂 Models loaded successfully!")

# ===================== Entry Point =====================

async def main():
    """Main entry point"""
    # Create configuration
    config = TradingConfig(
        binance_api_key=os.getenv('BINANCE_API_KEY', 'your_api_key'),
        binance_api_secret=os.getenv('BINANCE_API_SECRET', 'your_api_secret'),
        symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'LTCUSDT'],
        initial_balance=10000.0,
        paper_trading=True,  # Set to False for real trading
        update_interval=30,
        confidence_threshold=0.65
    )
    
    # Create and run bot
    bot = AdvancedTradingBot(config)
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
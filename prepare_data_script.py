# prepare_and_run.py - Complete Setup with Data Preparation
# -------------------------------------------------------------------
import os
import sys
import shutil
import subprocess
import pandas as pd

def prepare_data_files():
    """Prepare CSV files for training"""
    print("📊 Preparing data files...")
    
    # Check for CSV files in current directory or data folder
    csv_files = {
        'BTC': ['BTC_yfinance_daily.csv', 'data/BTC_yfinance_daily.csv', 'btc_yfinance_daily.csv'],
        'LTC': ['LTC_yfinance_daily.csv', 'data/LTC_yfinance_daily.csv', 'ltc_yfinance_daily.csv']
    }
    
    found_files = {}
    
    for symbol, possible_files in csv_files.items():
        for file_path in possible_files:
            if os.path.exists(file_path):
                found_files[symbol] = file_path
                print(f"✅ Found {symbol} data: {file_path}")
                break
    
    # Copy files to root directory if needed
    for symbol, source_path in found_files.items():
        target_path = f"{symbol}_yfinance_daily.csv"
        if source_path != target_path:
            try:
                shutil.copy2(source_path, target_path)
                print(f"✅ Copied {symbol} data to root directory")
            except Exception as e:
                print(f"❌ Error copying {symbol} data: {e}")
    
    # Check if we have necessary files
    missing = []
    for symbol in ['BTC', 'LTC']:
        if not os.path.exists(f"{symbol}_yfinance_daily.csv"):
            missing.append(symbol)
    
    if missing:
        print(f"\n❌ Missing data files for: {', '.join(missing)}")
        print("\n📥 Creating sample data files...")
        create_sample_data()
    
    return len(missing) == 0

def create_sample_data():
    """Create sample data files for testing"""
    import numpy as np
    from datetime import datetime, timedelta
    
    for symbol in ['BTC', 'LTC']:
        if not os.path.exists(f"{symbol}_yfinance_daily.csv"):
            print(f"📝 Creating sample {symbol} data...")
            
            # Generate sample data
            dates = []
            prices = []
            
            # Starting values
            if symbol == 'BTC':
                base_price = 30000
                volatility = 0.02
            else:
                base_price = 100
                volatility = 0.03
            
            # Generate 1000 days of data
            current_date = datetime.now() - timedelta(days=1000)
            current_price = base_price
            
            for i in range(1000):
                dates.append(current_date.strftime('%Y-%m-%d'))
                
                # Random walk
                change = np.random.normal(0, volatility)
                current_price *= (1 + change)
                
                # OHLCV data
                high = current_price * (1 + abs(np.random.normal(0, 0.005)))
                low = current_price * (1 - abs(np.random.normal(0, 0.005)))
                open_price = current_price * (1 + np.random.normal(0, 0.002))
                volume = np.random.randint(1000000, 10000000)
                
                prices.append({
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Open': round(open_price, 2),
                    'High': round(high, 2),
                    'Low': round(low, 2),
                    'Close': round(current_price, 2),
                    'Volume': volume
                })
                
                current_date += timedelta(days=1)
            
            # Save to CSV
            df = pd.DataFrame(prices)
            # Reorder columns to match expected format
            df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
            df.to_csv(f"{symbol}_yfinance_daily.csv", index=False)
            print(f"✅ Created {symbol}_yfinance_daily.csv with {len(df)} rows")

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'lightgbm',
        'python-binance',
        'requests'
    ]
    
    for package in packages:
        try:
            __import__(package.lower().replace('-', '_'))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📥 Installing {package}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            except:
                print(f"⚠️ Failed to install {package}")

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    dirs = [
        'ai_models',
        'ai_models/advanced_trading_ai',
        'logs',
        'data',
        'modules'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created {dir_path}")

def train_ai_model():
    """Train the AI model"""
    print("\n🧠 Training AI Model...")
    print("This may take 10-30 minutes depending on your computer")
    
    try:
        # Import and run trainer
        from ai_trainer import AdvancedAITrainer
        trainer = AdvancedAITrainer()
        success = trainer.train_complete_system()
        
        if success:
            print("✅ AI Model training completed!")
            return True
        else:
            print("❌ AI Model training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

def update_modules_safe():
    """Update module files with proper encoding"""
    print("\n📝 Updating module files...")
    
    # Create ai_predictor in modules if exists
    if os.path.exists('ai_predictor.py'):
        try:
            # Try UTF-8 first
            try:
                with open('ai_predictor.py', 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                # Fallback to latin-1
                with open('ai_predictor.py', 'r', encoding='latin-1') as f:
                    content = f.read()
            
            with open('modules/ai_predictor.py', 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Updated ai_predictor.py")
        except Exception as e:
            print(f"⚠️ Could not update ai_predictor.py: {e}")

def run_trading_bot():
    """Run the main trading bot"""
    print("\n🚀 Starting Trading Bot...")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, 'main.py'])
    except KeyboardInterrupt:
        print("\n👋 Trading bot stopped by user")
    except Exception as e:
        print(f"❌ Error running bot: {e}")

def main():
    """Main setup and run function"""
    print("🎯 Advanced AI Trading System Setup")
    print("=" * 50)
    
    # Step 1: Prepare data files
    data_ready = prepare_data_files()
    if not data_ready:
        print("\n⚠️ Using generated sample data for testing")
        print("💡 For real trading, use your actual historical data files")
    
    # Step 2: Install requirements
    install_requirements()
    
    # Step 3: Setup directories
    setup_directories()
    
    # Step 4: Check if model already exists
    if os.path.exists('ai_models/advanced_trading_ai/meta_model.pkl'):
        print("\n✅ AI Model already exists!")
        response = input("Do you want to retrain? (y/n): ")
        if response.lower() == 'y':
            train_ai_model()
    else:
        # Train model
        print("\n🆕 No existing model found. Training new model...")
        train_ai_model()
    
    # Step 5: Update modules
    update_modules_safe()
    
    # Step 6: Ask user what to do
    print("\n" + "=" * 50)
    print("✅ Setup completed!")
    print("\nWhat would you like to do?")
    print("1. Run trading bot now")
    print("2. Exit (run main.py manually later)")
    
    choice = input("\nEnter choice (1 or 2): ")
    
    if choice == '1':
        run_trading_bot()
    else:
        print("\n📌 To run the bot later, use: python main.py")
        print("💡 The AI system is now integrated and ready to use!")

if __name__ == "__main__":
    main()

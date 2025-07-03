# setup_and_run.py - Complete Setup and Execution Script
# -------------------------------------------------------------------
import os
import sys
import subprocess
import time

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'lightgbm',
        'TA-Lib',  # Note: This might need special installation
        'python-binance',
        'requests',
        'alpaca-trade-api'
    ]
    
    for package in packages:
        try:
            __import__(package.lower().replace('-', '_'))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📥 Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

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

def update_modules():
    """Update module files"""
    print("\n📝 Updating module files...")
    
# Copy ai_predictor.py to modules directory
if os.path.exists('ai_predictor.py'):
    with open('ai_predictor.py', 'r', encoding='utf-8', errors='replace') as src:
        content = src.read()
    with open('modules/ai_predictor.py', 'w', encoding='utf-8') as dst:
        dst.write(content)
    print("✅ Updated ai_predictor.py")

    
# Update ai_decider.py
if os.path.exists('ai_decider.py'):
    # อ่านไฟล์ต้นทาง
    with open('ai_decider.py', 'r', encoding='utf-8', errors='replace') as src:
        content = src.read()
        
   # สำรองไฟล์เดิม (modules/ai_decider.py -> modules/ai_decider_backup.py)
    if os.path.exists('modules/ai_decider.py'):
        with open('modules/ai_decider.py', 'r', encoding='utf-8', errors='replace') as original_file:
            original = original_file.read()
        with open('modules/ai_decider_backup.py', 'w', encoding='utf-8') as backup:
            backup.write(original)
        print("✅ Backed up original ai_decider.py")

        
    # เขียนไฟล์ใหม่
    with open('modules/ai_decider.py', 'w', encoding='utf-8') as dst:
        dst.write(content)
    print("✅ Updated ai_decider.py")

def run_trading_bot():
    """Run the main trading bot"""
    print("\n🚀 Starting Trading Bot...")
    print("=" * 50)
    
    try:
        # Run main.py
        subprocess.run([sys.executable, 'main.py'])
    except KeyboardInterrupt:
        print("\n👋 Trading bot stopped by user")
    except Exception as e:
        print(f"❌ Error running bot: {e}")

def main():
    """Main setup and run function"""
    print("🎯 Advanced AI Trading System Setup")
    print("=" * 50)
    
    # Step 1: Install requirements
    try:
        install_requirements()
    except Exception as e:
        print(f"⚠️ Some packages failed to install: {e}")
        print("Please install them manually")
    
    # Step 2: Setup directories
    setup_directories()
    
    # Step 3: Check if model already exists
    if os.path.exists('ai_models/advanced_trading_ai/meta_model.pkl'):
        print("\n✅ AI Model already exists!")
        response = input("Do you want to retrain? (y/n): ")
        if response.lower() == 'y':
            train_ai_model()
    else:
        # Train model
        print("\n🆕 No existing model found. Training new model...")
        train_ai_model()
    
    # Step 4: Update modules
    update_modules()
    
    # Step 5: Ask user what to do
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

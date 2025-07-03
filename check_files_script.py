# check_and_fix.py - ตรวจสอบและแก้ไขไฟล์
# -------------------------------------------------------------------
import os
import glob

def check_csv_files():
    """ตรวจสอบไฟล์ CSV ทั้งหมด"""
    print("🔍 ตรวจสอบไฟล์ CSV...")
    print("=" * 50)
    
    # หาไฟล์ CSV ทั้งหมด
    csv_files = []
    
    # ค้นหาในโฟลเดอร์ปัจจุบัน
    for file in glob.glob("*.csv"):
        csv_files.append(file)
        print(f"✅ พบไฟล์: {file}")
    
    # ค้นหาในโฟลเดอร์ data
    for file in glob.glob("data/*.csv"):
        csv_files.append(file)
        print(f"✅ พบไฟล์: {file}")
    
    # ตรวจสอบไฟล์ที่ต้องการ
    required_files = ['BTC_yfinance_daily.csv', 'LTC_yfinance_daily.csv']
    missing = []
    
    for req_file in required_files:
        found = False
        for csv_file in csv_files:
            if req_file.lower() in csv_file.lower():
                found = True
                print(f"\n📊 {req_file} -> พบที่: {csv_file}")
                
                # ถ้าไม่ได้อยู่ใน root ให้ copy มา
                if csv_file != req_file:
                    import shutil
                    try:
                        shutil.copy2(csv_file, req_file)
                        print(f"✅ คัดลอกไฟล์มาที่ root directory แล้ว")
                    except Exception as e:
                        print(f"❌ ไม่สามารถคัดลอกไฟล์: {e}")
                break
        
        if not found:
            missing.append(req_file)
    
    if missing:
        print(f"\n❌ ไม่พบไฟล์: {', '.join(missing)}")
    else:
        print("\n✅ พบไฟล์ครบทุกไฟล์!")
    
    return len(missing) == 0

def list_all_files():
    """แสดงไฟล์ทั้งหมดในโปรเจค"""
    print("\n📁 ไฟล์ทั้งหมดในโปรเจค:")
    print("=" * 50)
    
    # Python files
    print("\n🐍 Python Files:")
    for file in glob.glob("*.py"):
        size = os.path.getsize(file) / 1024  # KB
        print(f"  - {file} ({size:.1f} KB)")
    
    # CSV files
    print("\n📊 CSV Files:")
    for file in glob.glob("*.csv") + glob.glob("data/*.csv"):
        size = os.path.getsize(file) / 1024  # KB
        print(f"  - {file} ({size:.1f} KB)")
    
    # Directories
    print("\n📁 Directories:")
    for dir_name in ['modules', 'ai_models', 'logs', 'data']:
        if os.path.exists(dir_name):
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (not found)")

def quick_fix():
    """แก้ไขปัญหาอย่างรวดเร็ว"""
    print("\n🔧 Quick Fix...")
    
    # สร้างโฟลเดอร์ที่จำเป็น
    for folder in ['data', 'logs', 'modules', 'ai_models']:
        os.makedirs(folder, exist_ok=True)
    
    print("✅ สร้างโฟลเดอร์ที่จำเป็นแล้ว")

def main():
    print("🔍 ตรวจสอบระบบ Trading Bot")
    print("=" * 70)
    
    # แสดงไฟล์ทั้งหมด
    list_all_files()
    
    # ตรวจสอบไฟล์ CSV
    csv_ok = check_csv_files()
    
    # Quick fix
    quick_fix()
    
    print("\n" + "=" * 70)
    if csv_ok:
        print("✅ ระบบพร้อมใช้งาน!")
        print("\nขั้นตอนต่อไป:")
        print("1. รัน: python prepare_and_run.py")
        print("2. หรือรัน: python ai_trainer.py (ถ้ามี)")
    else:
        print("⚠️ ไม่พบไฟล์ CSV")
        print("\nวิธีแก้ไข:")
        print("1. ตรวจสอบว่าไฟล์ CSV อยู่ในโฟลเดอร์ที่ถูกต้อง")
        print("2. หรือรัน: python prepare_and_run.py (จะสร้างข้อมูลตัวอย่างให้)")

if __name__ == "__main__":
    main()

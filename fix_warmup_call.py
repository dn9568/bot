# fix_warmup_call.py - แก้ไขการเรียก warm_up_predictor

def fix_main_py():
    """แก้ไข main.py ให้เรียก warm_up_predictor หลังสร้าง predictor"""
    
    print("🔧 Fixing warm_up_predictor call in main.py...")
    
    # อ่าน main.py
    with open('main.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # หาบรรทัดที่สร้าง predictor
    for i, line in enumerate(lines):
        if 'predictor = AdvancedAIPredictor()' in line:
            print(f"Found predictor initialization at line {i+1}")
            
            # ตรวจสอบว่ามีการเรียก warm_up_predictor แล้วหรือไม่
            if i+1 < len(lines) and 'warm_up_predictor' not in lines[i+1]:
                # เพิ่มการเรียก warm_up_predictor
                indent = len(line) - len(line.lstrip())
                warm_up_line = ' ' * indent + 'warm_up_predictor(predictor)\n'
                lines.insert(i+1, warm_up_line)
                print(f"✅ Added warm_up_predictor call at line {i+2}")
                
                # บันทึกไฟล์
                with open('main.py', 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                print("✅ main.py updated successfully!")
                return True
            else:
                print("✅ warm_up_predictor call already exists")
                return False
    
    print("❌ Could not find predictor initialization line")
    return False

if __name__ == "__main__":
    print("🚀 Fixing warm_up_predictor call")
    print("="*50)
    
    if fix_main_py():
        print("\n✅ Fixed! Now run: python main.py")
        print("You should see historical data being loaded at startup")
    else:
        print("\n⚠️  No changes needed or error occurred")

# modules/alpaca_client.py
import alpaca_trade_api as tradeapi
import requests

def create_alpaca_clients(api_key, api_secret, base_url):
    """สร้าง Trading Client สำหรับ Alpaca"""
    try:
        # สร้าง REST client สำหรับทั้ง trading และ data
        client = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        
        # ทดสอบการเชื่อมต่อ
        account = client.get_account()
        print(f"✅ Alpaca Client connection successful. Account Status: {account.status}")
        
        # Return client 2 ตัวเพื่อให้ compatible กับ main.py (แต่จริงๆ เป็นตัวเดียวกัน)
        return client, client
        
    except Exception as e:
        print(f"❌ Failed to create Alpaca client: {e}")
        return None, None

def get_latest_prices(client, symbols):
    """ดึงราคาล่าสุดสำหรับหลายๆ Symbol จาก Alpaca"""
    prices = {}
    print("Fetching crypto prices using bars data...")
    
    # แมพ symbol จาก config format (BTCUSDT) เป็น Alpaca format (BTC/USD)
    symbol_mapping = {
        'BTCUSDT': 'BTC/USD',
        'ETHUSD': 'ETH/USD', 
        'BNBUSD': 'BNB/USD',
        'LTCUSD': 'LTC/USD',
        'BTCUSD': 'BTC/USD',
        'ETHUSDT': 'ETH/USD',
        'BNBUSDT': 'BNB/USD',
        'LTCUSDT': 'LTC/USD',
        'SOLUSD': 'SOL/USD',
        'SOLUSDT': 'SOL/USD'
    }
    
    
    for original_symbol in symbols:
        try:
            # แปลง symbol เป็น Alpaca format
            alpaca_symbol = symbol_mapping.get(original_symbol, original_symbol)
            if '/' not in alpaca_symbol and 'USD' in alpaca_symbol:
                # แปลง XXXUSD เป็น XXX/USD
                alpaca_symbol = alpaca_symbol.replace('USD', '/USD')
            
            price_found = False
            
            try:
                # ใช้ get_crypto_bars เพื่อดึงข้อมูลราคาล่าสุด
                bars = client.get_crypto_bars(
                    alpaca_symbol,
                    tradeapi.TimeFrame.Minute,
                    limit=1
                ).df
                
                if not bars.empty:
                    # ใช้ราคาปิดล่าสุด
                    price = float(bars['close'].iloc[-1])
                    prices[original_symbol] = price  # ใช้ original symbol เพื่อให้ตรงกับที่ config ต้องการ
                    print(f"✅ {original_symbol}: ${price:.2f}")
                    price_found = True
                    
            except Exception as e:
                print(f"⚠️  Could not get bars for {alpaca_symbol}: {e}")
            
            if not price_found:
                # ถ้ายังไม่ได้ราคา ลองใช้ REST API โดยตรง
                try:
                    headers = {
                        'APCA-API-KEY-ID': client._key_id,
                        'APCA-API-SECRET-KEY': client._secret_key
                    }
                    
                    # ลองดึงจาก crypto endpoint
                    url = f"{client._base_url}/v1beta3/crypto/us/latest/quotes?symbols={alpaca_symbol}"
                    response = requests.get(url, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'quotes' in data and alpaca_symbol in data['quotes']:
                            quote = data['quotes'][alpaca_symbol]
                            # ใช้ ask price หรือ bid price
                            price = float(quote.get('ap', quote.get('bp', 0)))
                            if price > 0:
                                prices[original_symbol] = price
                                print(f"✅ {original_symbol}: ${price:.2f}")
                                price_found = True
                    
                except Exception as e:
                    pass
            
            if not price_found:
                print(f"❌ Could not get price for {original_symbol}")
                # ใช้ราคา dummy เพื่อให้ bot ทำงานต่อได้ (เพื่อทดสอบ)
                dummy_prices = {
                    'BTCUSDT': 108000.0,
                    'BTCUSD': 108000.0,
                    'ETHUSD': 2500.0,
                    'ETHUSDT': 2500.0,
                    'BNBUSD': 400.0,
                    'BNBUSDT': 400.0,
                    'LTCUSD': 87.0,
                    'LTCUSDT': 87.0,
                    'SOLUSD': 180.0,
                    'SOLUSDT': 180.0
                }
                if original_symbol in dummy_prices:
                    prices[original_symbol] = dummy_prices[original_symbol]
                    print(f"⚠️  Using dummy price for {original_symbol}: ${prices[original_symbol]:.2f}")
                
        except Exception as e:
            print(f"❌ Error processing {original_symbol}: {e}")
    
    return prices
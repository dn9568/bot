# modules/binance_client.py

from binance.client import Client

def create_binance_data_client(api_key, api_secret):
    """
    สร้าง Binance REST client (Mainnet) สำหรับดึงราคาคริปโต
    """
    client = Client(api_key, api_secret)
    # ไม่ตั้ง testnet → เชื่อม Mainnet
    return client

def get_latest_prices(client, symbols):
    """
    คืน dict: { 'BTCUSDT': 108000.0, ... }
    """
    out = {}
    for s in symbols:
        tick = client.get_symbol_ticker(symbol=s)
        out[s] = float(tick['price'])
    return out

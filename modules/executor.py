# modules/executor.py

import logging

class BinancePaperExecutor:
    """
    Simulator for paper‐trade: บัญชีเสมือน ไม่ส่ง order จริง
    """

    def __init__(self, initial_balance: float, trade_amount: float):
        self.balance = initial_balance
        self.trade_amount = trade_amount
        self.positions = {}   # เช่น {'BTCUSDT': 0.01}
        logging.info(f"[Simulator] Init balance={self.balance:.2f}")

    def execute(self, action: int, symbol: str, price: float):
        """
        action: 1=BUY, -1=SELL, 0=HOLD
        """
        pos = self.positions.get(symbol, 0.0)

        if action == 1:
            qty = self.trade_amount / price
            self.positions[symbol] = pos + qty
            self.balance -= qty * price
            logging.info(f"➡️ BUY  {symbol}, qty={qty:.6f} @ {price:.2f}")

        elif action == -1 and pos > 0:
            # ขายทั้งหมด (close)  
            self.balance += pos * price
            logging.info(f"➡️ SELL {symbol}, qty={pos:.6f} @ {price:.2f}")
            self.positions[symbol] = 0.0

        else:
            logging.info(f"HOLD {symbol}")

        # คืนสถานะเงิน+position
        return {'balance': self.balance, 'position': self.positions.get(symbol, 0.0)}

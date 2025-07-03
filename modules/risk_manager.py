# risk_manager.py - Advanced Risk Management
# -------------------------------------------------------------------

class RiskManager:
    def __init__(self, max_risk_per_trade: float = 0.02, max_portfolio_risk: float = 0.1):
        self.max_risk_per_trade = max_risk_per_trade  # 2% per trade
        self.max_portfolio_risk = max_portfolio_risk  # 10% total portfolio risk
        self.current_risk = 0.0
        self.open_positions = {}
        
    def calculate_position_size(self, balance: float, entry_price: float, stop_loss: float) -> float:
        """Calculate optimal position size based on risk management"""
        if stop_loss <= 0:
            return balance * 0.01 / entry_price  # Conservative fallback
        
        risk_amount = balance * self.max_risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk > 0:
            position_size = risk_amount / price_risk
            return min(position_size, balance * 0.1 / entry_price)  # Max 10% of balance
        else:
            return balance * 0.01 / entry_price
    
    def should_allow_trade(self, proposed_risk: float) -> bool:
        """Check if trade should be allowed based on risk limits"""
        if self.current_risk + proposed_risk > self.max_portfolio_risk:
            return False
        return True
    
    def update_risk(self, trade_risk: float):
        """Update current portfolio risk"""
        self.current_risk += trade_risk
        self.current_risk = max(0, self.current_risk)  # Can't be negative


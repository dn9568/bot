# portfolio_manager.py - Advanced Portfolio Management
# -------------------------------------------------------------------

class AdvancedPortfolioManager:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}  # {symbol: quantity}
        self.trade_history = []
        self.daily_returns = []
        self.max_balance = initial_balance
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
    def execute_trade(self, symbol: str, action: int, price: float, trade_amount: float) -> Dict:
        """Execute trade with advanced portfolio management"""
        
        if action == 0:  # Hold
            return {
                'balance': self.balance,
                'position': self.positions.get(symbol, 0),
                'action': 'HOLD'
            }
        
        # Position sizing based on Kelly Criterion (simplified)
        max_position_size = self.balance * 0.1  # Max 10% per position
        trade_amount = min(trade_amount, max_position_size)
        
        # Calculate quantity
        if action == 1:  # Buy
            if self.balance >= trade_amount:
                quantity = trade_amount / price
                self.balance -= trade_amount
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': price,
                    'amount': trade_amount
                }
                self.trade_history.append(trade_record)
                self.total_trades += 1
                
                return {
                    'balance': self.balance,
                    'position': self.positions[symbol],
                    'action': 'BUY',
                    'quantity': quantity
                }
        
        elif action == -1:  # Sell
            current_position = self.positions.get(symbol, 0)
            if current_position > 0:
                sell_quantity = min(current_position, trade_amount / price)
                sell_amount = sell_quantity * price
                
                self.balance += sell_amount
                self.positions[symbol] -= sell_quantity
                
                # Calculate P&L for this trade
                # (Simplified - in real system would track individual trade costs)
                
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': sell_quantity,
                    'price': price,
                    'amount': sell_amount
                }
                self.trade_history.append(trade_record)
                self.total_trades += 1
                
                return {
                    'balance': self.balance,
                    'position': self.positions[symbol],
                    'action': 'SELL',
                    'quantity': sell_quantity
                }
        
        return {
            'balance': self.balance,
            'position': self.positions.get(symbol, 0),
            'action': 'HOLD'
        }
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value including positions"""
        total_value = self.balance
        
        for symbol, quantity in self.positions.items():
            if symbol in current_prices and quantity > 0:
                total_value += quantity * current_prices[symbol]
        
        return total_value
    
    def get_performance_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate advanced performance metrics"""
        current_value = self.calculate_portfolio_value(current_prices)
        
        # Update max balance and drawdown
        if current_value > self.max_balance:
            self.max_balance = current_value
        
        current_drawdown = (self.max_balance - current_value) / self.max_balance * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Calculate returns
        total_return = (current_value - self.initial_balance) / self.initial_balance * 100
        
        # Win rate (simplified)
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades * 100
        
        return {
            'current_value': current_value,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'positions': dict(self.positions)
        }

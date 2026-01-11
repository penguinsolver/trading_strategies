"""
Trading cost model for realistic backtesting.
"""
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    """Order execution type."""
    MARKET = "market"  # Taker
    LIMIT = "limit"    # Maker


@dataclass
class CostModel:
    """
    Comprehensive trading cost model.
    
    Includes:
    - Trading fees (maker/taker)
    - Slippage estimate
    - Funding rate estimate
    """
    
    maker_fee: float = 0.0001      # 0.01% (1 bp)
    taker_fee: float = 0.00035     # 0.035% (3.5 bp)
    slippage_bps: float = 1.0      # 1 basis point slippage
    funding_rate_8h: float = 0.0001  # 0.01% per 8h
    
    def get_fee_rate(self, order_type: OrderType = OrderType.MARKET) -> float:
        """Get fee rate based on order type."""
        if order_type == OrderType.LIMIT:
            return self.maker_fee
        return self.taker_fee
    
    def calculate_fee(
        self,
        notional: float,
        order_type: OrderType = OrderType.MARKET,
    ) -> float:
        """
        Calculate fee for a trade.
        
        Args:
            notional: Trade notional value (price × size)
            order_type: Market (taker) or limit (maker)
            
        Returns:
            Fee amount in USD
        """
        return notional * self.get_fee_rate(order_type)
    
    def calculate_slippage(self, price: float) -> float:
        """
        Calculate slippage-adjusted price.
        
        For entries, this increases the entry price (worse fill).
        For exits, this should be applied in the opposite direction.
        
        Args:
            price: Base price
            
        Returns:
            Price impact from slippage in USD per unit
        """
        return price * (self.slippage_bps / 10000)
    
    def apply_entry_slippage(self, price: float, side: str) -> float:
        """
        Apply slippage to entry price.
        
        For longs: price increases (worse fill)
        For shorts: price decreases (worse fill)
        """
        slippage = self.calculate_slippage(price)
        if side.lower() == "long":
            return price + slippage
        else:
            return price - slippage
    
    def apply_exit_slippage(self, price: float, side: str) -> float:
        """
        Apply slippage to exit price.
        
        For longs: price decreases (worse fill)
        For shorts: price increases (worse fill)
        """
        slippage = self.calculate_slippage(price)
        if side.lower() == "long":
            return price - slippage
        else:
            return price + slippage
    
    def calculate_funding(
        self,
        notional: float,
        hours_held: float,
        side: str,
    ) -> float:
        """
        Estimate funding cost/credit for holding a position.
        
        Simplification: Assumes constant positive funding rate
        (longs pay shorts). In reality, this fluctuates.
        
        Args:
            notional: Position notional value
            hours_held: Duration of position in hours
            side: "long" or "short"
            
        Returns:
            Funding cost (positive = cost, negative = credit)
        """
        # Convert 8h rate to hourly
        hourly_rate = self.funding_rate_8h / 8
        
        # Total funding for the holding period
        funding_amount = notional * hourly_rate * hours_held
        
        # Longs pay funding (cost), shorts receive (credit)
        if side.lower() == "long":
            return funding_amount  # Cost
        else:
            return -funding_amount  # Credit
    
    def total_trade_cost(
        self,
        entry_notional: float,
        exit_notional: float,
        hours_held: float,
        side: str,
        entry_order_type: OrderType = OrderType.MARKET,
        exit_order_type: OrderType = OrderType.MARKET,
    ) -> dict:
        """
        Calculate total costs for a round-trip trade.
        
        Returns:
            Dictionary with breakdown of costs
        """
        entry_fee = self.calculate_fee(entry_notional, entry_order_type)
        exit_fee = self.calculate_fee(exit_notional, exit_order_type)
        
        # Slippage on both sides
        avg_notional = (entry_notional + exit_notional) / 2
        slippage_cost = avg_notional * (self.slippage_bps / 10000) * 2  # Entry + exit
        
        funding = self.calculate_funding(avg_notional, hours_held, side)
        
        return {
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "total_fees": entry_fee + exit_fee,
            "slippage": slippage_cost,
            "funding": funding,
            "total_cost": entry_fee + exit_fee + slippage_cost + max(0, funding),
        }

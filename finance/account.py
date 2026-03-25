from dataclasses import dataclass, field
from finance.ohlcv import OHLCV
import numpy as np
from typing import Optional
import pandas as pd

@dataclass
class PortfolioFeatures:
    fiat_ratio: float
    crypto_ratio: float
    log_return_total: float
    log_return_step: float
    @classmethod
    def initial_value(cls):
        return cls(fiat_ratio=1.0, crypto_ratio=0.0, log_return_total=0.0, log_return_step=0.0)
    def to_numpy(self) -> np.ndarray:
        return np.array([self.fiat_ratio, self.crypto_ratio, self.log_return_total, self.log_return_step])

@dataclass
class AccountState:
    reward: float
    terminated: bool
    portfolio_features: PortfolioFeatures
    
@dataclass
class Account:
    """Account class for managing trading account.
    """
    initial_balance: float
    fee_open_percent: float = 0.001
    fee_close_percent: float = 0.001
    adjust_impact_coeff: float = 0.05  # Coefficient to adjust market impact cost
    noise_stddev: float = 0.05  # Standard deviation for stochastic slippage
    balance: float = field(init=False, default=0.0)
    crypto_quantity: float = field(init=False, default=0.0)
    equity: float = field(init=False, default=0.0)
    fee_open_total: float = field(init=False, default=0.0)
    fee_close_total: float = field(init=False, default=0.0)
    total_trades: int = field(init=False, default=0)
    np_random: np.random.Generator = field(init=False)
    prev_equity: float = field(init=False)
    prices: list[dict[str, float]] = field(init=False, default_factory=list)
    counter: int = field(init=False, default=0)
    
    def __post_init__(self):
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.prev_equity = self.initial_balance
    
    def reset(self, np_random: np.random.Generator):
        """Reset the account to its initial state.
        """
        self.np_random = np_random
        self.balance = self.initial_balance
        self.crypto_quantity = 0.0
        self.equity = self.initial_balance
        self.fee_open_total = 0.0
        self.fee_close_total = 0.0
        self.total_trades = 0
        if self.counter > 0:
            self.write_prices_to_csv()
        self.prices = []
        self.counter += 1

    def write_prices_to_csv(self):
        """Write the account data to a CSV file.
        """
        filename = f"prices_{self.counter}.csv"
        df = pd.DataFrame(self.prices)
        df.to_csv(filename, index=False)

    def step(self, a: float, ohlcv: OHLCV):
        """interface for agent to perform an order

        Args:
            a (float): action of agent, range [-1, 1], negative for sell, positive for buy
            ohlcv (OHLCV): The OHLCV data for the current period.
            price_close (float): The closing price of the asset.
        """
        price_ideal = ohlcv.open  # Using open price as the ideal price for the trade
        price_close = ohlcv.close  # Using close price as the closing price for the trade
        self.prev_equity = self.equity

        price_entry = price_ideal

        if a > 0:  # Buy
            amount_to_spend = self.balance * a
            slippage_cost = self._slippage_and_market_impact_cost(price_ideal, amount_to_spend, ohlcv)
            price_entry = price_ideal + slippage_cost
            fee_open = amount_to_spend * self.fee_open_percent
            self.fee_open_total += fee_open
            quantity_received = (amount_to_spend - fee_open) / price_entry
            self.balance -= amount_to_spend
            self.crypto_quantity += quantity_received
            if quantity_received > 0:
                self.total_trades += 1
            self.prices.append({"timestamp": ohlcv.timestamp, "action": a, "price_slippage": price_entry, "price_ideal": price_ideal, "price_close": price_close})
        elif a < 0:  # Sell
            crypto_sell = self.crypto_quantity * a * -1
            balance_ideal = crypto_sell * price_ideal
            slippage_cost = self._slippage_and_market_impact_cost(price_ideal, -balance_ideal, ohlcv)
            price_exit = price_ideal - slippage_cost
            balance_received = crypto_sell * price_exit
            fee_close = balance_received * self.fee_close_percent
            balance_received -= fee_close
            self.fee_close_total += fee_close
            self.balance += balance_received
            self.crypto_quantity -= crypto_sell
            if balance_received > 0:
                self.total_trades += 1
            self.prices.append({"timestamp": ohlcv.timestamp, "action": a, "price_slippage": price_exit, "price_ideal": price_ideal, "price_close": price_close})
        else:
            self.prices.append({"timestamp": ohlcv.timestamp, "action": a, "price_slippage": price_ideal, "price_ideal": price_ideal, "price_close": price_close})
        
        # Update equity after the trade
        self.equity = self.balance + self.crypto_quantity * price_close
        reward = self.reward()
        terminated = self.equity <= self.initial_balance * 0.1
        portfolio_features = PortfolioFeatures(
            fiat_ratio=self.balance / self.initial_balance,
            crypto_ratio=self.crypto_quantity * price_close / self.initial_balance,
            log_return_total=np.log(self.equity / self.initial_balance),
            log_return_step=np.log(self.equity / self.prev_equity)
        )


        return AccountState(
            reward=reward,
            terminated=terminated,
            portfolio_features=portfolio_features
        )

    def reward(self) -> float:
        r = (self.equity - self.prev_equity) / self.prev_equity
        return r
        
    def get_equity(self) -> float:
        return self.equity
    
    def get_stats(self) -> dict:
        return {
            'balance': self.balance,
            'crypto_quantity': self.crypto_quantity,
            'equity': self.equity,
            'fee_open_total': self.fee_open_total,
            'fee_close_total': self.fee_close_total,
            'total_trades': self.total_trades
        }

    def _slippage_and_market_impact_cost(self, price_ideal: float, estimate_volume: float, ohlcv: OHLCV) -> float:
        """Calculate the slippage and market impact cost based on the trade size.

        Args:
            price_ideal (float): The ideal price of the asset.
            estimate_volume (float): An estimate of the slippage cost for a full trade.
            ohlcv (OHLCV): The OHLCV data for the current period.
        """
        if estimate_volume > 0:
            # Calculate the slippage cost based on the trade size
            range_slippage = ohlcv.high - price_ideal
        else:
            range_slippage = price_ideal - ohlcv.low
        
        deterministic_slippage = np.sqrt(abs(estimate_volume) / (ohlcv.volume + 1e-6)) 
        
        stochastic_slippage = self.np_random.uniform(0, self.noise_stddev)

        slippage_coefficient = np.minimum(1.0, self.adjust_impact_coeff * deterministic_slippage + stochastic_slippage)
        
        return range_slippage * slippage_coefficient
    
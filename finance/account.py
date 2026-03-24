from dataclasses import dataclass, field
from finance.ohlcv import OHLCV
import numpy as np
from typing import Optional

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
    adjust_impact_coeff: float = 0.1  # Coefficient to adjust market impact cost
    noise_stddev: float = 0.05  # Standard deviation for stochastic slippage
    balance: float = field(init=False, default=0.0)
    crypto_quantity: float = field(init=False, default=0.0)
    equity: float = field(init=False, default=0.0)
    fee_open_total: float = field(init=False, default=0.0)
    fee_close_total: float = field(init=False, default=0.0)
    total_trades: int = field(init=False, default=0)
    np_random: np.random.Generator = field(init=False)
    prev_equity: float = field(init=False)
    
    def __post_init__(self):
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.prev_equity = self.initial_balance
        self.np_random = np.random.default_rng()
    
    def reset(self, seed: Optional[int] = None):
        """Reset the account to its initial state.
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.balance = self.initial_balance
        self.crypto_quantity = 0.0
        self.equity = self.initial_balance
        self.fee_open_total = 0.0
        self.fee_close_total = 0.0
        self.total_trades = 0
    
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

        if a > 0:  # Buy
            amount_to_spend = self.balance * a
            slippage_cost = self._slippage_and_market_impact_cost(price_ideal, amount_to_spend, ohlcv)
            price_entry = price_ideal + slippage_cost
            fee_open = amount_to_spend * self.fee_open_percent
            self.fee_open_total += fee_open
            quantity_received = (amount_to_spend - fee_open) / price_entry
            self.balance -= amount_to_spend
            self.crypto_quantity += quantity_received
            self.total_trades += 1
        elif a < 0:  # Sell
            sign_quantity = self.crypto_quantity * a
            crypto_to_sell = sign_quantity * -1
            slippage_cost = self._slippage_and_market_impact_cost(price_ideal, sign_quantity, ohlcv)
            price_exit = price_ideal - slippage_cost
            fee_close = crypto_to_sell * price_exit * self.fee_close_percent
            self.fee_close_total += fee_close
            balance_received = crypto_to_sell * price_exit * (1 - self.fee_close_percent)
            self.balance += balance_received
            self.crypto_quantity -= crypto_to_sell
            self.total_trades += 1

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

        slippage_coefficient = np.minimum(1.0, self.adjust_impact_coeff * (deterministic_slippage + stochastic_slippage))
        
        return range_slippage * slippage_coefficient
    
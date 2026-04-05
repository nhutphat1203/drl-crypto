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
    slippage_rate: float = 0.0003
    threshold_rebalance: float = 0.05
    balance: float = field(init=False, default=0.0)
    crypto_quantity: float = field(init=False, default=0.0)
    equity: float = field(init=False, default=0.0)
    fee_open_total: float = field(init=False, default=0.0)
    fee_close_total: float = field(init=False, default=0.0)
    total_buy: int = field(init=False, default=0)
    total_sell: int = field(init=False, default=0)
    np_random: np.random.Generator = field(init=False)
    prev_equity: float = field(init=False)
    total_slippage: float = field(init=False, default=0.0)
    debugs: list = field(init=False, default_factory=list)
    counter_reset: int = field(init=False, default=0)
    prev_price: Optional[float] = field(init=False, default=None)
    action_mean : float = field(init=False, default=0.0)
    no_rebalance_count: int = field(init=False, default=0)
    
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
        self.prev_equity = self.initial_balance
        self.fee_open_total = 0.0
        self.fee_close_total = 0.0
        self.total_buy = 0
        self.total_sell = 0
        self.total_slippage = 0.0
        self.counter_reset += 1
        self.debugs = []
        self.prev_price = None
        self.action_mean = 0.0
        self.no_rebalance_count = 0

    def step(self, a: float, ohlcv: OHLCV):
        """interface for agent to perform an order

        Args:
            a (float): action of agent, range [0, 1]
            ohlcv (OHLCV): The OHLCV data for the current period.
            price_close (float): The closing price of the asset.
        """
        
        self.action_mean = 0.9 * self.action_mean + 0.1 * a
        
        price_ideal = ohlcv.open  # Using open price as the ideal price for the trade
        price_close = ohlcv.close  # Using close price as the closing price for the trade
        self.prev_equity = self.equity
        price_execute = price_ideal
        
        threshold_rebalance = self.threshold_rebalance
        crypto_value = self.crypto_quantity * price_ideal
        cash_value = self.balance
        total_value = crypto_value + cash_value
        crypto_portfolio_weight = crypto_value / total_value
        sign_delta = a - crypto_portfolio_weight
        delta = abs(sign_delta)
        if delta >= threshold_rebalance:
            if sign_delta > 0:
                # Need to buy crypto
                amount_cash_buy = delta * total_value
                price_execute = price_ideal * (1 + self.slippage_rate)
                quantity = amount_cash_buy / price_execute
                fee_open = quantity * self.fee_open_percent
                quantity_received = quantity - fee_open
                self.balance -= amount_cash_buy
                self.crypto_quantity += quantity_received
                self.fee_open_total += fee_open * price_execute
                self.total_buy += 1
                self.total_slippage += (price_execute - price_ideal) * quantity
            elif sign_delta < 0:
                # Need to sell crypto
                amount_crypto_sell = delta * total_value / price_ideal
                price_execute = price_ideal * (1 - self.slippage_rate)
                balance_received = amount_crypto_sell * price_execute
                fee_close = balance_received * self.fee_close_percent
                balance_received -= fee_close
                self.balance += balance_received
                self.crypto_quantity -= amount_crypto_sell
                self.fee_close_total += fee_close
                self.total_sell += 1
                self.total_slippage += (price_ideal - price_execute) * amount_crypto_sell
        else:
            self.no_rebalance_count += 1
        
        if self.prev_price is None:
            self.prev_price = ohlcv.open
        # Update equity after the trade
        self.equity = self.balance + self.crypto_quantity * price_close
        reward = self.reward(price_close, self.prev_price)
        terminated = self.equity <= self.initial_balance * 0.1
        fiat_ratio = self.balance / self.equity
        crypto_ratio = 1 - fiat_ratio
        portfolio_features = PortfolioFeatures(
            fiat_ratio=fiat_ratio,
            crypto_ratio=crypto_ratio,
            log_return_total=np.log(self.equity / self.initial_balance),
            log_return_step=np.log(self.equity / self.prev_equity),
        )
        self.prev_price = price_close
        return AccountState(
            reward=reward,
            terminated=terminated,
            portfolio_features=portfolio_features
        )

    def reward(self, price, prev_price) -> float:
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
            'total_buy': self.total_buy,
            'total_sell': self.total_sell,
            'total_slippage': self.total_slippage
        }

    def get_final_stats(self) -> str:
        total_return = self.equity / self.initial_balance - 1
        total_fee = self.fee_open_total + self.fee_close_total
        profit = self.equity - self.initial_balance
        slippage_fee_ratio = self.total_slippage / total_fee if total_fee > 0 else 0

        lines = [
            "----------------------------------------------------------",
            "                  Final Trading Stats",
            "----------------------------------------------------------",
            f"portfolio/initial_balance   | {self.initial_balance:.6f}",
            f"portfolio/final_equity      | {self.equity:.6f}",
            f"portfolio/profit            | {profit:.6f}",
            f"portfolio/total_return      | {total_return:.6f}",
            f"trade/total_buy             | {self.total_buy}",
            f"trade/total_sell            | {self.total_sell}",
            f"cost/total_fee_buy          | {self.fee_open_total:.6f}",
            f"cost/total_fee_sell         | {self.fee_close_total:.6f}",
            f"cost/total_fee              | {total_fee:.6f}",
            f"cost/slippage_fee_ratio     | {slippage_fee_ratio:.4f}",
            f"cost/total_slippage         | {self.total_slippage:.6f}",
            f"action/action_mean_ema      | {self.action_mean:.4f}",
            f"action/no_rebalance_count   | {self.no_rebalance_count}",
            "----------------------------------------------------------",
        ]
        return "\n".join(lines)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional
from data_manager.episode import Episode, StepData
from data_manager.data_provider import DataProvider
from finance.account import Account, AccountState, PortfolioFeatures

class Market(gym.Env):
    def __init__(self, df: pd.DataFrame,
                name: str,
                initial_balance: float,
                window_size: int, 
                episode_length: int, 
                fee_rate_open: float = 0.01,
                fee_rate_close: float = 0.01,
                test_mode: bool = False,
                verbose: int = 0):
        super().__init__()
        self.name = name
        self.df = df
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.episode_length = episode_length
        self.test_mode = test_mode
        self.fee_rate_open = fee_rate_open  
        self.fee_rate_close = fee_rate_close
        self.verbose = verbose
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.provider = DataProvider(data=self.df, window_size=self.window_size, tick_for_episode=self.episode_length, use_full_for_one_episode=self.test_mode)
        cols_to_exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        num_features = len([col for col in self.df.columns if col not in cols_to_exclude]) 
        self.observation_space = spaces.Dict({
            'time_series': spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, num_features), dtype=np.float32),
            'portfolio_features': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32) # Giả sử có 4 features
        })
        self.account = Account(initial_balance=self.initial_balance, 
                                fee_open_percent=self.fee_rate_open,
                                fee_close_percent=self.fee_rate_close)
        self.episode: Episode = None
        self.current_data: StepData = None

    def _obs(self, data: StepData, portfolio_features: PortfolioFeatures) -> dict:
        return {
            'time_series': data.observation,
            'portfolio_features': portfolio_features.to_numpy()
        }
    def _info(self) -> dict:
        return self.account.get_stats()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        self.provider.reset(np_random=self.np_random)
        self.account.reset(np_random=self.np_random)
        self.episode = self.provider.next_episode()
        self.current_data = self.episode.next()
        portfolio_features = PortfolioFeatures.initial_value()
        return self._obs(self.current_data, portfolio_features), self._info()

    def step(self, action: np.float32):
        scalar_action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        account_state: AccountState = self.account.step(a=scalar_action, ohlcv=self.current_data.ohlcv)
        terminated = account_state.terminated
        truncated = self.current_data.no_more_data
        reward = account_state.reward
        if not (terminated or truncated):
            self.current_data = self.episode.next()
        obs = self._obs(self.current_data, account_state.portfolio_features)
        info = self._info()
        info['timestamp'] = self.current_data.ohlcv.timestamp
        info['price'] = self.current_data.ohlcv.close
        info['action'] = scalar_action
        info['reward'] = reward

        if (terminated or truncated) and self.verbose > 0:
            print(f"Name: {self.name}")
            print(self.account.get_final_stats())

        return obs, reward, terminated, truncated, info
    
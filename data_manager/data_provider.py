from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from data_manager.episode import Episode
from typing import Optional

@dataclass
class DataProvider:
    data: pd.DataFrame
    window_size: int
    tick_for_episode: int
    use_full_for_one_episode: bool = field(default=False)
    current_index: int = field(init=False, default=0)
    list_indices: list[int] = field(init=False, default_factory=list)
    generator: np.random.Generator = field(init=False)

    def __post_init__(self):
        data_length = len(self.data)
        if self.use_full_for_one_episode:
            self.tick_for_episode = data_length
        if self.window_size > data_length:
            raise ValueError("window_size is greater than data length")
        if self.window_size > self.tick_for_episode:
            raise ValueError("window_size is greater than tick_for_episode")
        if self.tick_for_episode > data_length:
            raise ValueError("tick_for_episode is greater than data length")
        self.list_indices = list(range(self.tick_for_episode - 1, data_length))
        self.generator = None

    def reset(self, np_random: np.random.Generator):
        flag = False
        if self.generator is None:
            flag = True
        self.generator = np_random
        if flag:
            self.generator.shuffle(self.list_indices)

    def next_episode(self) -> Episode:
        if self.current_index >= len(self.list_indices):
            self.generator.shuffle(self.list_indices)
            self.current_index = 0
        last_index = self.list_indices[self.current_index]
        data = self.data.iloc[last_index - self.tick_for_episode + 1 : last_index + 1]
        self.current_index += 1
        return Episode(data=data, window_size=self.window_size)
"""Uniform replay, deliberately simple; full stacks trade RAM for readability."""
from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.items = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, observation, action, reward, next_observation, terminated, truncated):
        # Own every image: a native emulator view must not rewrite past experiences.
        self.items.append(Transition(observation.copy(), action, reward,
                                     next_observation.copy(), terminated, truncated))

    def sample(self, size: int) -> list[Transition]:
        return self.rng.sample(list(self.items), size)

    def __len__(self):
        return len(self.items)

"""Uniform replay in preallocated arrays: a fixed-size circular memory.

Each slot holds one transition (state, action, reward, next state, flags).
Writing reuses the oldest slot once the memory is full, like ``deque(maxlen)``.
Sampling builds a batch with one NumPy indexing operation per field instead of
stacking thousands of small Python objects. Full 4-frame stacks are stored for
readability, so memory is about ``capacity × 56 KB`` (20.000 → ~1,13 GB).
"""
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Batch:
    observations: np.ndarray       # (B, 4, 84, 84) uint8
    actions: np.ndarray            # (B,) int64
    rewards: np.ndarray            # (B,) float32
    next_observations: np.ndarray  # (B, 4, 84, 84) uint8
    terminated: np.ndarray         # (B,) bool
    truncated: np.ndarray          # (B,) bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.size = 0      # Stored transitions, never more than capacity.
        self.position = 0  # Slot for the next write; once full, also the oldest.
        self.observations = None  # Allocated on first add, from its shape/dtype.

    def _allocate(self, observation: np.ndarray) -> None:
        shape = (self.capacity, *observation.shape)
        self.observations = np.zeros(shape, observation.dtype)
        self.next_observations = np.zeros(shape, observation.dtype)
        self.actions = np.zeros(self.capacity, np.int64)
        self.rewards = np.zeros(self.capacity, np.float32)
        self.terminated = np.zeros(self.capacity, bool)
        self.truncated = np.zeros(self.capacity, bool)

    def add(self, observation, action, reward, next_observation, terminated, truncated):
        if self.observations is None:
            self._allocate(observation)
        slot = self.position
        # Assignment copies the pixels: a native emulator view mutating later
        # can never rewrite a stored experience.
        self.observations[slot] = observation
        self.next_observations[slot] = next_observation
        self.actions[slot] = action
        self.rewards[slot] = reward
        self.terminated[slot] = terminated
        self.truncated[slot] = truncated
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, size: int) -> Batch:
        # Choose ages (0 = oldest) exactly as random.sample on a deque would,
        # then translate ages into circular slots.
        ages = np.array(self.rng.sample(range(self.size), size))
        oldest = self.position if self.size == self.capacity else 0
        slots = (oldest + ages) % self.capacity
        return Batch(self.observations[slots], self.actions[slots], self.rewards[slots],
                     self.next_observations[slots], self.terminated[slots],
                     self.truncated[slots])

    def __len__(self):
        return self.size

"""One decision repeats an action, then adds one grayscale frame to the history."""
from collections import deque

import gymnasium as gym
import numpy as np
from PIL import Image

from .config import EnvironmentConfig
from .environment import REWARD_VERSION


class DecisionEnvironment(gym.Wrapper):
    def __init__(self, env, config: EnvironmentConfig):
        super().__init__(env)
        self.config = config
        self.observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
        self.history = deque(maxlen=4)
        self.decisions = 0
        self.raw_frames = 0
        self.ended = False

    @staticmethod
    def grayscale(frame):
        image = Image.fromarray(frame).convert('L').resize((84, 84), Image.Resampling.BILINEAR)
        return np.asarray(image).copy()

    def reset(self, *, seed=None, options=None):
        frame, info = self.env.reset(seed=seed, options=options)
        gray = self.grayscale(frame)
        self.history.clear()
        for _ in range(4):
            self.history.append(gray.copy())
        self.decisions = 0
        self.raw_frames = 0
        self.ended = False
        return np.stack(self.history), self._with_counters(info, 0)

    def step(self, action):
        if self.ended:
            raise RuntimeError('El episodio terminó; reinícialo antes de avanzar.')
        reward_sum = 0.0
        components: dict[str, float] = {}
        executed = 0
        for _ in range(self.config.action_repeat):
            frame, reward, terminated, truncated, info = self.env.step(action)
            reward_sum += reward
            executed += 1
            for name, value in info.get('reward_components', {}).items():
                components[name] = components.get(name, 0.0) + value
            if terminated or truncated:
                break
        self.raw_frames += executed
        self.decisions += 1
        if self.decisions >= self.config.max_episode_steps and not terminated:
            truncated = True
        self.ended = terminated or truncated
        self.history.append(self.grayscale(frame))
        info = dict(info)
        info['reward_components'] = components
        return np.stack(self.history), reward_sum, terminated, truncated, self._with_counters(info, executed)

    def _with_counters(self, info, executed):
        return {**info, 'decisions': self.decisions, 'raw_frames': self.raw_frames,
                'frames_this_decision': executed}

    def render(self):
        return np.array(self.env.render(), copy=True)

    def identity(self):
        return {'mode': 'demo' if self.config.demo else 'nes', 'level': '1-1',
                'rom_sha256': getattr(self.env, 'rom_sha256', None),
                'action_repeat': self.config.action_repeat,
                'max_episode_steps': self.config.max_episode_steps,
                'reward_version': 'synthetic-v1' if self.config.demo else REWARD_VERSION,
                'reset_version': 'synthetic-v1' if self.config.demo else 'engine-clock-v2',
                'observation_shape': [4, 84, 84]}

"""Authored synthetic fixture, NOT Mario, NOT NES. Never selected implicitly."""
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

from .environment import ACTION_NAMES


class SyntheticPlatformer(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = gym.spaces.Box(0, 255, (240, 256, 3), np.uint8)
        self.render_mode = 'rgb_array'
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 18
        self.maximum = self.x
        self.frames = 0
        self.jump = 0
        self.done = False
        self.success = False
        self.death = False
        self._draw()
        return self.frame.copy(), self._info()

    def step(self, action):
        if self.done:
            raise RuntimeError('Synthetic episode ended; reset required.')
        if not self.action_space.contains(action):
            raise ValueError('Invalid synthetic action.')
        old_max = self.maximum
        self.frames += 1
        if action in (2, 4, 5) and self.jump == 0:
            self.jump = 16
        jumping = self.jump > 0
        self.jump = max(0, self.jump - 1)
        if action in (1, 2, 3, 4):
            self.x += 3 if action in (3, 4) else 2
        elif action == 6:
            self.x = max(10, self.x - 2)
        self.maximum = max(self.maximum, self.x)
        self.death = 108 <= self.x <= 124 and not jumping
        self.success = self.x >= 230
        self.done = self.death or self.success
        reward = self.maximum - old_max - .01
        reward += 15 if self.success else (-15 if self.death else 0)
        self._draw()
        return self.frame.copy(), float(reward), self.done, False, self._info()

    def _info(self):
        return {'x_pos': self.x, 'progress_max': self.maximum,
                'flag_get': self.success, 'death': self.death,
                'time': max(0, 400 - self.frames // 24), 'world': 1, 'stage': 1,
                'raw_frames': self.frames, 'fixture': True}

    def _draw(self):
        image = Image.new('RGB', (256, 240), '#101b31')
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), 'SYNTHETIC FIXTURE / NOT MARIO', fill='#ffd585')
        draw.text((10, 28), f'frame {self.frames:04d}   x {self.x:03d}', fill='#c7d4e8')
        for x in range(0, 256, 32):
            draw.line((x, 50, x, 216), fill='#182841')
        draw.rectangle((0, 216, 107, 240), fill='#314864')
        draw.rectangle((125, 216, 256, 240), fill='#314864')
        draw.line((232, 92, 232, 216), fill='#8be2c3', width=2)
        draw.polygon([(232, 92), (248, 98), (232, 107)], fill='#8be2c3')
        height = int(35 * np.sin(self.jump / 16 * np.pi)) if self.jump else 0
        color = '#ff7183' if self.death else '#a895ff'
        draw.rounded_rectangle((self.x - 6, 199 - height, self.x + 6, 214 - height),
                               radius=3, fill=color)
        if self.done:
            draw.text((80, 120), 'SUCCESS' if self.success else 'FAILURE', fill=color)
        self.frame = np.asarray(image).copy()

    def render(self):
        return self.frame.copy()

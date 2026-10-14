"""Real NES integration: external ROM only, one life, first level, no ROM patches.

RAM semantics are documented in docs/ARCHITECTURE.md. This intentionally implements
only SMB1 1-1; it is not a general-purpose NES game environment.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import gymnasium as gym
import numpy as np

from .config import EnvironmentConfig

# NES controller masks: right=bit 7, left=bit 6, B=bit 1, A=bit 0.
ACTION_MASKS = (0, 128, 129, 130, 131, 1, 64)
ACTION_NAMES = ('Nada', 'Derecha', 'Derecha + salto', 'Derecha + correr',
                'Derecha + correr + salto', 'Salto', 'Izquierda')
REWARD_VERSION = 'progress-time-terminal-v1'


class MarioNES(gym.Env):
    """Expose copied RGB frames and RAM-derived reward/metrics from a legal ROM."""

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}

    def __init__(self, rom: str):
        from nes_py import NESEnv  # Optional native dependency; demo does not need it.

        path = Path(rom).expanduser().resolve()
        data = path.read_bytes()
        if len(data) < 16 or data[:4] != b'NES\x1a':
            raise ValueError('ROM inválida: se necesita un archivo iNES .nes de SMB1.')
        mapper = (data[6] >> 4) | (data[7] & 0xf0)
        if mapper != 0 or data[4] != 2 or data[5] != 1:
            raise ValueError('Solo se admite SMB1 estándar NROM (32 KiB PRG, 8 KiB CHR), '
                             'no otros juegos, hacks ni ROMs modificadas.')
        self.rom_sha256 = hashlib.sha256(data).hexdigest()
        self.emulator = NESEnv(str(path), render_mode='rgb_array')
        self.action_space = gym.spaces.Discrete(len(ACTION_MASKS))
        self.observation_space = gym.spaces.Box(0, 255, (240, 256, 3), np.uint8)
        self.render_mode = 'rgb_array'
        self._start_snapshot = None
        self._frame = np.zeros((240, 256, 3), np.uint8)
        self._done = False
        self._max_x = 0
        self._last_time = 0
        self._components: dict[str, float] = {}

    def _decimal(self, start: int, count: int) -> int:
        value = 0
        for digit in self.emulator.ram[start:start + count]:
            value = value * 10 + int(digit)
        return value

    def _metrics(self) -> dict:
        ram = self.emulator.ram
        state = int(ram[0x000e])
        death = state in (0x06, 0x0b) or int(ram[0x00b5]) > 1
        flag_present = any(int(enemy) == 0x31 for enemy in ram[0x0016:0x001b])
        success = (flag_present and int(ram[0x001d]) == 3) or int(ram[0x0770]) == 2
        x = int(ram[0x006d]) * 256 + int(ram[0x0086])
        return {'x_pos': x, 'progress_max': self._max_x, 'flag_get': success,
                'death': death, 'time': self._decimal(0x07f8, 3),
                'world': int(ram[0x075f]) + 1, 'stage': int(ram[0x075c]) + 1,
                'player_state': state, 'reward_components': dict(self._components)}

    def _start_level(self) -> None:
        # Bounded boot avoids hanging indefinitely on the wrong ROM.
        # START is pulsed, never held; stop before it can pause active gameplay.
        for frame in range(3600):
            button = 8 if frame >= 60 and frame % 30 == 0 else 0
            self.emulator.step(button)
            ram = self.emulator.ram
            if (int(ram[0x0770]) == 1 and int(ram[0x0772]) == 3 and int(ram[0x000e]) == 8
                    and self._decimal(0x07f8, 3) > 0):
                if int(ram[0x075f]) != 0 or int(ram[0x075c]) != 0:
                    raise ValueError('La ROM no inició en el nivel 1-1 esperado.')
                # The game mode changes while the title's player state is stale.
                # OperMode_Task=3 means the new level engine has initialized.
                # Wait for the first game-clock tick before freezing reset state.
                initial_time = self._decimal(0x07f8, 3)
                for _ in range(120):
                    self.emulator.step(0)
                    if self._decimal(0x07f8, 3) < initial_time and int(ram[0x000e]) == 8:
                        # The first OAM/PPU update trails the clock initialization.
                        # Four idle frames give the initial screenshot its sprites.
                        for _ in range(4):
                            self.emulator.step(0)
                        return
                raise ValueError('El juego no avanzó tras START; revisa la ROM.')
        raise ValueError('No se pudo iniciar SMB1 1-1 en 3600 frames. '
                         'Comprueba que la ROM sea estándar y compatible.')

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._start_snapshot is None:
            self.emulator.reset(seed=seed)
            self._start_level()
            self._start_snapshot = self.emulator.dump_state()
        else:
            self.emulator.load_state(self._start_snapshot)
        self._done = False
        self._components = {}
        metrics = self._metrics()
        self._max_x = metrics['x_pos']
        self._last_time = metrics['time']
        self._frame = np.array(self.emulator.render(), copy=True)
        return self._frame.copy(), self._metrics()

    def step(self, action):
        if self._done:
            raise RuntimeError('El episodio terminó; llama reset antes de step.')
        if not self.action_space.contains(action):
            raise ValueError(f'Acción inválida: {action}')
        frame, _, _, _, _ = self.emulator.step(ACTION_MASKS[int(action)])
        metrics = self._metrics()
        progress = max(0, metrics['x_pos'] - self._max_x)
        self._max_x = max(self._max_x, metrics['x_pos'])
        # Ignore teleport-like jumps; reward only new ground, not back-and-forth.
        self._components = {
            'progress': float(progress if progress <= 5 else 0),
            'time': float(min(0, metrics['time'] - self._last_time)),
            'death': -25.0 if metrics['death'] else 0.0,
            'completion': 50.0 if metrics['flag_get'] else 0.0,
        }
        self._last_time = metrics['time']
        self._done = metrics['death'] or metrics['flag_get']
        self._frame = np.array(frame, copy=True)  # Native views mutate on the next step.
        reward = float(np.clip(sum(self._components.values()), -15, 15))
        return self._frame.copy(), reward, self._done, False, self._metrics()

    def render(self):
        return self._frame.copy()

    def close(self):
        self.emulator.close()


def make_environment(config: EnvironmentConfig):
    """Return the same preprocessed contract for NES and an explicit test fixture."""
    from .preprocessing import DecisionEnvironment

    config.validate()
    if config.demo:
        from .fixture import SyntheticPlatformer
        raw = SyntheticPlatformer()
    else:
        try:
            raw = MarioNES(config.rom)
        except ImportError as error:
            raise ValueError('Falta nes-py: instala con `uv sync --extra nes`.') from error
    return DecisionEnvironment(raw, config)

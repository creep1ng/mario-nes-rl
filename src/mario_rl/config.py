"""Configuration stays in plain dataclasses, not hidden framework defaults."""
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentConfig:
    demo: bool = False
    rom: str | None = None
    action_repeat: int = 4
    max_episode_steps: int = 3000  # Agent decisions, not emulator frames.

    def validate(self) -> None:
        if self.action_repeat < 1 or self.max_episode_steps < 1:
            raise ValueError('action_repeat y max_episode_steps deben ser positivos.')
        if self.demo and self.rom:
            raise ValueError('Elige --demo o --rom, no ambos.')
        if not self.demo and not self.rom:
            raise ValueError('Falta ROM: proporciona --rom /ruta/legal/super-mario-bros.nes; '
                             '--demo es solo un entorno sintético de prueba.')
        if self.rom and not Path(self.rom).is_file():
            raise ValueError(f'ROM no encontrada: {self.rom}')


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1_000_000
    seed: int = 7
    batch_size: int = 32
    replay_capacity: int = 20_000
    warmup: int = 5000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 200_000
    learn_every: int = 4
    target_every: int = 1000  # Optimizer updates.
    eval_every: int = 25_000
    eval_episodes: int = 5
    checkpoint_every: int = 25_000
    device: str = 'cpu'
    envs: int = 1  # Parallel emulators, one process each. 1 = original loop.

    def validate(self) -> None:
        positive = ['steps', 'batch_size', 'replay_capacity', 'epsilon_decay',
                    'learn_every', 'target_every', 'eval_every', 'eval_episodes',
                    'checkpoint_every', 'envs']
        for field in positive:
            if getattr(self, field) < 1:
                raise ValueError(f'{field} debe ser positivo.')
        if self.warmup < 0 or self.replay_capacity < self.batch_size:
            raise ValueError('warmup debe ser >= 0 y replay_capacity >= batch_size.')
        if not 0 <= self.gamma <= 1 or self.learning_rate <= 0:
            raise ValueError('gamma debe estar entre 0 y 1; learning_rate debe ser > 0.')
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError('Se requiere 0 <= epsilon_end <= epsilon_start <= 1.')
        if self.device != 'cpu' and not self.device.startswith('cuda'):
            raise ValueError('device debe ser cpu o cuda[:índice].')

    def epsilon(self, step: int) -> float:
        if step < self.warmup:
            return 1.0
        fraction = min((step - self.warmup) / self.epsilon_decay, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def to_dict(self) -> dict:
        return asdict(self)

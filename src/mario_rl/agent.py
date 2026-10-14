"""Action selection, Bellman update, target synchronization and safe checkpoints."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
from pathlib import Path
import pickle
import random

import numpy as np
import torch
from torch import nn

from .config import TrainConfig
from .environment import ACTION_NAMES
from .model import QNetwork
from .replay import ReplayBuffer


class DQNAgent:
    def __init__(self, config: TrainConfig):
        config.validate()
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA no está disponible; usa --device cpu.')
        self.rng = random.Random(config.seed)
        # Creating an evaluation model must not change a training model's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.online = QNetwork().to(self.device)
            self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.updates = 0

    def q_values(self, observation) -> np.ndarray:
        with torch.no_grad():
            pixels = torch.as_tensor(observation, device=self.device).unsqueeze(0)
            return self.online(pixels)[0].cpu().numpy()

    def select_action(self, observation, epsilon: float = 0.0) -> int:
        if epsilon > 0 and self.rng.random() < epsilon:
            return self.rng.randrange(len(ACTION_NAMES))
        return int(self.q_values(observation).argmax())

    def learn(self, replay: ReplayBuffer) -> float:
        batch = replay.sample(self.config.batch_size)
        states = torch.as_tensor(np.stack([t.observation for t in batch]), device=self.device)
        following = torch.as_tensor(np.stack([t.next_observation for t in batch]), device=self.device)
        actions = torch.tensor([t.action for t in batch], device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        terminals = torch.tensor([t.terminated for t in batch], dtype=torch.float32, device=self.device)
        predicted = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            # External truncation is not a terminal state: retain its bootstrap.
            future = self.target(following).max(dim=1).values
            expected = rewards + self.config.gamma * (1 - terminals) * future
        loss = nn.functional.smooth_l1_loss(predicted, expected)
        if not torch.isfinite(loss):
            raise RuntimeError('Pérdida no finita; se detiene el entrenamiento.')
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: Path, step: int, environment: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {'format_version': 1, 'online': self.online.state_dict(),
                      'target': self.target.state_dict(), 'optimizer': self.optimizer.state_dict(),
                      'config': asdict(self.config), 'environment': environment,
                      'actions': list(ACTION_NAMES), 'step': step, 'updates': self.updates}
        temporary = path.with_suffix('.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)  # Readers never observe a partially written checkpoint.

    @staticmethod
    def read_checkpoint(path: Path, device: str = 'cpu') -> dict:
        """Read once per call; fingerprint exactly the bytes that were decoded."""
        try:
            payload = Path(path).read_bytes()
            data = torch.load(io.BytesIO(payload), map_location=device, weights_only=True)
            if not isinstance(data, dict) or not isinstance(data.get('environment'), dict):
                raise ValueError('Contenido del checkpoint inválido.')
            if data['format_version'] != 1 or data['actions'] != list(ACTION_NAMES):
                raise ValueError('Formato o espacio de acciones incompatible.')
            TrainConfig(**data['config']).validate()
            for name in ('step', 'updates'):
                if type(data[name]) is not int or data[name] < 0:
                    raise ValueError(f'Contador inválido: {name}.')
            data['_source_sha256'] = hashlib.sha256(payload).hexdigest()
            return data
        except (OSError, EOFError, pickle.UnpicklingError, KeyError, TypeError, RuntimeError, ValueError) as error:
            raise ValueError(f'No se pudo cargar el checkpoint {path}: {error}') from error

    @classmethod
    def load(cls, path: Path, environment: dict, device: str = 'cpu',
             *, training_config: TrainConfig | None = None):
        try:
            data = cls.read_checkpoint(path, device)
            for key in ('mode', 'rom_sha256', 'action_repeat', 'reward_version', 'reset_version', 'observation_shape'):
                if data['environment'].get(key) != environment[key]:
                    raise ValueError(f'Entorno incompatible con checkpoint: {key}.')
            config = training_config or TrainConfig(**{**data['config'], 'device': device})
            agent = cls(config)
            agent.online.load_state_dict(data['online'])
            agent.target.load_state_dict(data['target'])
            agent.updates = data['updates']
            if training_config is not None:
                agent.optimizer.load_state_dict(data['optimizer'])
                # CLI overrides are explicit; Adam moments/step remain restored.
                for group in agent.optimizer.param_groups:
                    group['lr'] = config.learning_rate
            agent.online.train(training_config is not None)
            return agent, {'step': data['step'], 'updates': data['updates'],
                           'environment': data['environment'],
                           'source_sha256': data['_source_sha256']}
        except (OSError, EOFError, pickle.UnpicklingError, KeyError, TypeError, RuntimeError, ValueError) as error:
            raise ValueError(f'No se pudo cargar el checkpoint {path}: {error}') from error

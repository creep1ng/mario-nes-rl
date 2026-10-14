"""Frozen evaluation policies share a small interface; none can learn here."""
from pathlib import Path
import random

from .agent import DQNAgent
from .config import TrainConfig
from .environment import ACTION_NAMES


class RandomPolicy:
    label = 'Aleatorio'

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def select_action(self, observation):
        return self.rng.randrange(len(ACTION_NAMES))

    def q_values(self, observation):
        return None


class RightPolicy:
    label = 'Baseline: correr a la derecha'

    def select_action(self, observation):
        return 3

    def q_values(self, observation):
        return None


class NetworkPolicy:
    def __init__(self, agent: DQNAgent, label: str):
        self.agent = agent
        self.label = label
        self.agent.online.eval()

    def select_action(self, observation):
        return self.agent.select_action(observation, epsilon=0.0)

    def q_values(self, observation):
        return self.agent.q_values(observation).tolist()


def make_policy(spec: str, identity: dict, seed: int, checkpoints: list[Path]):
    if spec == 'random':
        return RandomPolicy(seed)
    if spec == 'right':
        return RightPolicy()
    if spec == 'untrained':
        return NetworkPolicy(DQNAgent(TrainConfig(seed=seed)), 'DQN sin entrenar')
    if spec.startswith('checkpoint:'):
        try:
            index = int(spec.split(':')[1])
            if index < 0:
                raise ValueError('Índice negativo.')
            path = checkpoints[index]
        except (ValueError, IndexError) as error:
            raise ValueError('Checkpoint desconocido.') from error
        agent, metadata = DQNAgent.load(path, identity)
        return NetworkPolicy(agent, f'{path.stem} · {metadata["step"]:,} decisiones')
    raise ValueError(f'Política desconocida: {spec}')

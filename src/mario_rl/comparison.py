"""Independent environments and frozen policies. UI reads never advance time."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from .config import EnvironmentConfig
from .environment import ACTION_NAMES, make_environment
from .metrics import runtime_metadata, write_json
from .policies import make_policy
from .recording import VideoRecorder, mosaic


@dataclass
class Panel:
    environment: object
    policy: object
    spec: str
    observation: object = None
    info: dict | None = None
    reward: float = 0.0
    steps: int = 0
    done: bool = False
    terminated: bool = False
    truncated: bool = False
    action: int | None = None
    q: list | None = None
    frame: str = ''

    def refresh_image(self):
        buffer = BytesIO()
        Image.fromarray(self.environment.render()).save(buffer, format='PNG')
        self.frame = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()

    def reset(self, seed):
        self.observation, self.info = self.environment.reset(seed=seed)
        self.reward = 0.0
        self.steps = 0
        self.done = self.terminated = self.truncated = False
        self.action = None
        self.q = self.policy.q_values(self.observation)
        self.refresh_image()

    def step(self):
        if self.done:
            return
        self.q = self.policy.q_values(self.observation)
        self.action = self.policy.select_action(self.observation)
        self.observation, reward, self.terminated, self.truncated, self.info = self.environment.step(self.action)
        self.reward += reward
        self.steps += 1
        self.done = self.terminated or self.truncated
        self.refresh_image()

    def state(self):
        status = 'éxito' if self.info['flag_get'] else 'muerte' if self.info['death'] else 'límite' if self.truncated else 'listo' if not self.steps else 'jugando'
        return {'policy': self.spec, 'label': self.policy.label, 'frame': self.frame,
                'steps': self.steps, 'raw_frames': self.info['raw_frames'],
                'reward': self.reward, 'progress_max': self.info['progress_max'],
                'action': None if self.action is None else ACTION_NAMES[self.action],
                'action_index': self.action, 'q_values': self.q,
                'done': self.done, 'terminated': self.terminated,
                'truncated': self.truncated, 'success': bool(self.info['flag_get']),
                'death': bool(self.info['death']), 'status': status}


class ComparisonRunner:
    def __init__(self, config: EnvironmentConfig, checkpoints: list[Path],
                 output: Path, seed: int = 7, policies: list[str] | None = None):
        config.validate()
        self.config = config
        self.seed = seed
        self.checkpoints = [path.resolve() for path in checkpoints]
        for path in self.checkpoints:
            if not path.is_file():
                raise ValueError(f'Checkpoint no encontrado: {path}')
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.tick = 0
        self.panels: list[Panel] = []
        self.recorder = None
        self.last_recording = None
        self.runtime = runtime_metadata()
        self.catalog = [{'id': 'random', 'label': 'Aleatorio'},
                        {'id': 'right', 'label': 'Correr a la derecha'},
                        {'id': 'untrained', 'label': 'DQN sin entrenar'}]
        self.catalog += [{'id': f'checkpoint:{index}', 'label': path.name}
                         for index, path in enumerate(self.checkpoints)]
        specs = policies or ['random', 'right', 'checkpoint:0' if checkpoints else 'untrained']
        if not 1 <= len(specs) <= 4:
            raise ValueError('Se admiten entre 1 y 4 paneles.')
        try:
            for index, spec in enumerate(specs):
                environment = make_environment(config)
                # Register ownership before loading: errors still close the emulator.
                panel = Panel(environment, None, spec)
                self.panels.append(panel)
                policy = self._policy(spec, environment.identity(), index)
                panel.policy = policy
                panel.reset(seed)
        except Exception:
            self.close()
            raise
        self.identity = self.panels[0].environment.identity()

    def _policy(self, spec, identity, index):
        if spec not in {item['id'] for item in self.catalog}:
            raise ValueError('Política desconocida; elige una opción del catálogo.')
        return make_policy(spec, identity, self.seed + index, self.checkpoints)

    def snapshot(self):
        return {'mode': 'demo' if self.config.demo else 'nes', 'seed': self.seed,
                'tick': self.tick, 'environment': self.identity,
                'catalog': self.catalog, 'panels': [panel.state() for panel in self.panels],
                'recording': self.recorder is not None,
                'recording_frames': 0 if self.recorder is None else self.recorder.frames,
                'last_recording': self.last_recording,
                'all_done': all(panel.done for panel in self.panels)}

    def step(self):
        if all(panel.done for panel in self.panels):
            return self.snapshot()
        for panel in self.panels:
            panel.step()
        self.tick += 1
        if self.recorder:
            self.recorder.append(mosaic(self.panels, self.config.demo, self.tick))
            if self.recorder.frames >= 1800:
                self.stop_recording()
        return self.snapshot()

    def reset(self, seed=None):
        if self.recorder:
            raise ValueError('Detén la grabación antes de reiniciar o cambiar agentes.')
        if seed is not None:
            if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
                raise ValueError('Seed debe ser un entero entre 0 y 2147483647.')
            self.seed = seed
        for index, panel in enumerate(self.panels):
            panel.policy = self._policy(panel.spec, panel.environment.identity(), index)
            panel.reset(self.seed)
        self.tick = 0
        return self.snapshot()

    def select(self, index, spec):
        if type(index) is not int or not 0 <= index < len(self.panels):
            raise ValueError('Panel inválido.')
        if self.recorder:
            raise ValueError('Detén la grabación antes de cambiar agentes.')
        self._policy(spec, self.identity, index)  # Validate before mutating the comparison.
        self.panels[index].spec = spec
        return self.reset()

    def report(self):
        data = self.snapshot()
        for panel in data['panels']:
            panel.pop('frame')
        return {**data, 'runtime': self.runtime,
                'checkpoints': [str(path) for path in self.checkpoints],
                'warning': 'Synthetic fixture; NOT Mario or NES' if self.config.demo else None,
                'recording_semantics': 'one visual frame per agent decision; no audio'}

    def start_recording(self, fps=15):
        if self.recorder:
            raise ValueError('Ya hay una grabación activa.')
        if type(fps) is not int or not 1 <= fps <= 60:
            raise ValueError('fps debe ser un entero entre 1 y 60.')
        frame = mosaic(self.panels, self.config.demo, self.tick)
        self.recorder = VideoRecorder(self.output / 'recordings', *frame.size, fps)
        self.recorder.append(frame)
        return self.snapshot()

    def stop_recording(self):
        if not self.recorder:
            if self.last_recording:
                return self.last_recording
            raise ValueError('No hay una grabación activa.')
        recorder, self.recorder = self.recorder, None
        self.last_recording = recorder.close()
        write_json(recorder.path.with_suffix('.json'), self.report())
        return self.last_recording

    def close(self):
        try:
            if self.recorder:
                self.stop_recording()
        finally:
            for panel in self.panels:
                panel.environment.close()

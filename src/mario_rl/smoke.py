"""Behavior-first pipeline verification. Produces inspectable frames + report."""
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from .config import EnvironmentConfig
from .environment import make_environment
from .metrics import runtime_metadata, write_json


def smoke(config: EnvironmentConfig, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    environment = make_environment(config)
    report = {'environment': environment.identity(), 'runtime': runtime_metadata(),
              'real_mario': not config.demo, 'success_seen': False, 'death_seen': False,
              'truncation_seen': False, 'repeat_stops_at_end': False,
              'frames_are_independent': False, 'scenarios': []}
    try:
        observation, _ = environment.reset(seed=7)
        before = environment.render()
        saved = before.copy()
        Image.fromarray(before).save(output / 'initial.png')
        actions = [4, 3, 0] if config.demo else [3, 4]
        for scenario, action in enumerate(actions):
            observation, info = environment.reset(seed=7)
            assert observation.shape == (4, 84, 84) and observation.dtype == np.uint8
            assert np.array_equal(observation[0], observation[-1])
            limit = 100 if not config.demo else 3 if action == 0 else 100
            if config.demo and action == 0:
                environment.close()
                environment = make_environment(replace(config, max_episode_steps=limit))
                observation, info = environment.reset(seed=7)
            terminated = truncated = False
            for _ in range(limit):
                observation, reward, terminated, truncated, info = environment.step(action)
                assert np.isfinite(reward)
                assert observation.shape == (4, 84, 84)
                if terminated or truncated:
                    break
            report['success_seen'] |= bool(info['flag_get'])
            report['death_seen'] |= bool(info['death'])
            report['truncation_seen'] |= truncated
            if terminated and info['frames_this_decision'] < config.action_repeat:
                report['repeat_stops_at_end'] = True
            if terminated or truncated:
                try:
                    environment.step(action)
                    raise AssertionError('Stepping a finished episode was allowed')
                except RuntimeError:
                    pass
            report['scenarios'].append({'action': action, 'terminated': terminated,
                                        'truncated': truncated, 'info': info})
            Image.fromarray(environment.render()).save(output / f'scenario-{scenario}.png')
        report['frames_are_independent'] = bool(np.array_equal(before, saved))
        report['status'] = 'passed synthetic behavior checks' if config.demo else 'NES interaction executed; not a learning benchmark'
        write_json(output / 'report.json', report)
        return report
    finally:
        environment.close()

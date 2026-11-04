"""CLI boundaries keep experiments reproducible and the browser optional."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import sys

import torch

from .config import EnvironmentConfig, TrainConfig
from .environment import make_environment
from .metrics import runtime_metadata, write_json
from .policies import make_policy


def build_parser():
    parser = argparse.ArgumentParser(description='DQN académico + visor local de Mario NES.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'evaluate', 'compare', 'smoke'):
        subparser = subparsers.add_parser(command)
        source = subparser.add_mutually_exclusive_group()
        source.add_argument('--rom', help='Ruta a ROM legal SMB1 estándar .nes. No se descarga ninguna ROM.')
        source.add_argument('--demo', action='store_true', help='Fixture sintético; NO es Mario ni NES.')
        subparser.add_argument('--action-repeat', type=int, default=None if command == 'train' else 4)
        subparser.add_argument('--max-episode-steps', type=int, default=None if command == 'train' else 3000)
        subparser.add_argument('--seed', type=int, default=None if command == 'train' else 7)
        subparser.add_argument('--output', type=Path)
        subparser.add_argument('--torch-threads', type=int, default=2,
                               help='Evita sobreparalelizar batches pequeños en CPU.')
        if command == 'train':
            # None distinguishes an explicit CLI override from an inherited value.
            fields = ('steps', 'warmup', 'batch_size', 'replay_capacity', 'epsilon_decay',
                      'learn_every', 'target_every', 'eval_every', 'eval_episodes', 'checkpoint_every',
                      'envs')
            helps = {'envs': 'Emuladores en paralelo, un proceso y núcleo de CPU cada uno '
                             '(1 = bucle original). Recomendado: núcleos − 2.'}
            for field in fields:
                subparser.add_argument('--' + field.replace('_', '-'), type=int, default=None,
                                       help=helps.get(field))
            for field in ('gamma', 'learning_rate', 'epsilon_start', 'epsilon_end'):
                subparser.add_argument('--' + field.replace('_', '-'), type=float, default=None)
            subparser.add_argument('--device', default=None)
            subparser.add_argument('--resume', type=Path,
                                   help='Continúa desde un checkpoint en una salida NUEVA; --steps es el objetivo total.')
            subparser.add_argument('--no-progress', action='store_true',
                                   help='Desactiva la barra tqdm; conserva mensajes y archivos de métricas.')
        elif command == 'evaluate':
            subparser.add_argument('--checkpoint', type=Path)
            subparser.add_argument('--policy', choices=['random', 'right', 'untrained'], default='random')
            subparser.add_argument('--episodes', type=int, default=10)
        elif command == 'compare':
            subparser.add_argument('--checkpoint', type=Path, action='append', default=[],
                                   help='Repetible; checkpoints explícitos, sin explorar rutas externas.')
            subparser.add_argument('--agents', nargs='+', help='random right untrained checkpoint:0 ... (1–4)')
            subparser.add_argument('--port', type=int, default=8765)
    return parser


def training_configuration(args):
    """Resume inherits configuration; fresh runs keep the usual defaults."""
    from .agent import DQNAgent

    saved = DQNAgent.read_checkpoint(args.resume) if args.resume else {}
    defaults = TrainConfig()
    values = {}
    for name in TrainConfig.__dataclass_fields__:
        explicit = getattr(args, name)
        values[name] = (explicit if explicit is not None else
                        saved.get('config', {}).get(name, getattr(defaults, name)))
    config = TrainConfig(**values)
    config.validate()
    cores = os.cpu_count() or 1
    if config.envs >= cores:
        print(f'Aviso: --envs {config.envs} con {cores} núcleos deja sin CPU libre a la red '
              f'y al sistema; suele rendir más con {max(1, cores - 2)}.', file=sys.stderr)
    if saved and config.steps <= saved['step']:
        raise ValueError(f'--steps es un objetivo total y debe ser mayor que '
                         f'las {saved["step"]} decisiones del checkpoint.')
    old_environment = saved.get('environment', {})
    environment = EnvironmentConfig(
        args.demo, args.rom,
        args.action_repeat if args.action_repeat is not None else old_environment.get('action_repeat', 4),
        args.max_episode_steps if args.max_episode_steps is not None else old_environment.get('max_episode_steps', 3000))
    return config, environment


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.torch_threads < 1:
            raise ValueError('torch-threads debe ser positivo.')
        torch.set_num_threads(args.torch_threads)
        if args.command == 'train':
            config, environment = training_configuration(args)
        else:
            environment = EnvironmentConfig(args.demo, args.rom, args.action_repeat, args.max_episode_steps)
        environment.validate()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        if args.command == 'train':
            from .training import train
            output = args.output or Path('runs') / f'train-{stamp}'
            result = train(config, environment, output, show_progress=not args.no_progress,
                           resume=args.resume)
        elif args.command == 'evaluate':
            from .evaluation import evaluate
            if args.episodes < 1:
                raise ValueError('episodes debe ser positivo.')
            env = make_environment(environment)
            try:
                identity = env.identity()
                checkpoints = [args.checkpoint] if args.checkpoint else []
                policy = make_policy('checkpoint:0' if checkpoints else args.policy,
                                     identity, args.seed, checkpoints)
            finally:
                env.close()
            result = {**evaluate(policy, environment, args.episodes, args.seed),
                      'runtime': runtime_metadata()}
            write_json(args.output or Path('runs') / f'evaluation-{stamp}.json', result)
        elif args.command == 'smoke':
            from .smoke import smoke
            result = smoke(environment, args.output or Path('artifacts') / f'smoke-{stamp}')
        else:
            from .comparison import ComparisonRunner
            from .server import serve
            if not 0 <= args.port <= 65535:
                raise ValueError('port debe estar entre 0 y 65535.')
            runner = ComparisonRunner(environment, args.checkpoint,
                                      args.output or Path('runs') / f'viewer-{stamp}',
                                      args.seed, args.agents)
            # Ensure recordings are finalized when a local process manager stops us.
            def stop(signum, frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, stop)
            serve(runner, args.port)
            return
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError) as error:
        parser.exit(2, f'Error: {error}\n')
    except KeyboardInterrupt:
        print('\nEjecución detenida.', file=sys.stderr)
        raise SystemExit(130)

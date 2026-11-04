"""Readable training loop: interact → remember → learn → evaluate → checkpoint.

With ``envs=N`` every round asks the agent for N actions at once, advances the
N emulators concurrently and then handles each transition as one decision, in
environment order. Learning, checkpoints and CSV rows still count decisions, so
``envs=1`` is exactly the original single-emulator loop.
"""
from dataclasses import asdict
import csv
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

from .agent import DQNAgent
from .config import EnvironmentConfig, TrainConfig
from .evaluation import evaluate
from .metrics import EpisodeResult, runtime_metadata, write_json
from .parallel import make_environments
from .policies import NetworkPolicy
from .replay import ReplayBuffer


def train(config: TrainConfig, environment_config: EnvironmentConfig, output: Path,
          *, show_progress: bool = True, resume: Path | None = None):
    config.validate()
    environment_config.validate()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'El directorio {output} no está vacío. Usa una ruta nueva; '
                         'no se sobrescriben experimentos. Para continuar usa '
                         '--resume CHECKPOINT y una salida nueva.')
    environments = make_environments(environment_config, config.envs)
    agent = None
    output_ready = False
    step = 0
    episode = 0
    latest_loss = None
    best_score = (-1.0, -1.0)
    def save_checkpoint(filename: str) -> None:
        path = output / filename
        agent.save(path, step, identity)
        # Keep stdout reserved for the CLI's JSON result; do not break the bar.
        tqdm.write(f'Checkpoint: {path.resolve()} (decisiones={step:,})', file=sys.stderr)

    try:
        identity = environments.identity()
        resume_info = None
        if resume is not None:
            agent, restored = DQNAgent.load(resume, identity, config.device, training_config=config)
            step = restored['step']
            if config.steps <= step:
                raise ValueError(f'--steps debe ser mayor que {step}; es el objetivo total.')
            resume_info = {'path': str(resume.resolve()), 'sha256': restored['source_sha256'],
                           'start_step': step, 'start_updates': agent.updates,
                           'replay_reset': True, 'episode_reset': True, 'rng_restored': False,
                           'replay_refill_decisions': config.warmup}
            tqdm.write(f'Reanudación: decisión={step:,} updates={agent.updates} '
                       f'epsilon={config.epsilon(step):.3f}. Replay/partida/RNG nuevos; '
                       f'warmup={config.warmup} decisiones nuevas antes de aprender.', file=sys.stderr)
        else:
            agent = DQNAgent(config)
        start_step = step
        replay = ReplayBuffer(config.replay_capacity, config.seed)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'config.json', {'training': asdict(config),
                   'environment': identity, 'runtime': runtime_metadata(), 'resume': resume_info})
        output_ready = True
        if config.envs > 1:
            tqdm.write(f'Emuladores paralelos: {config.envs} procesos. '
                       f'Dispositivo de la red: {config.device}.', file=sys.stderr)
        save_checkpoint(f'step-{step:09d}.pt')
        with tqdm(total=config.steps, initial=start_step, desc='Entrenando', unit='decisión',
                  dynamic_ncols=True, mininterval=0.5,
                  disable=not show_progress, file=sys.stderr) as progress, \
             (output / 'episodes.csv').open('w', newline='') as episodes_file, \
             (output / 'updates.csv').open('w', newline='') as updates_file, \
             (output / 'evaluation.csv').open('w', newline='') as evaluation_file:
            episode_writer = csv.DictWriter(episodes_file, fieldnames=[
                'episode', 'global_step', 'epsilon', 'seed', 'reward', 'progress_max',
                'decisions', 'raw_frames', 'success', 'death', 'truncated', 'budget_end', 'env'])
            update_writer = csv.DictWriter(updates_file, fieldnames=['step', 'update', 'loss', 'epsilon'])
            evaluation_writer = csv.DictWriter(evaluation_file, fieldnames=[
                'step', 'success_rate', 'mean_reward', 'mean_progress'])
            for writer in (episode_writer, update_writer, evaluation_writer):
                writer.writeheader()
            # Per-environment state. Episode k always uses seed + k, whichever
            # emulator plays it; with one environment this is the old numbering.
            count = environments.count
            episode_ids = list(range(count))
            episode_seeds = [config.seed + index for index in range(count)]
            episodes_started = count
            observations, infos = [], []
            for index in range(count):
                observation, info = environments.reset(index, seed=episode_seeds[index])
                observations.append(observation)
                infos.append(info)
            totals = [0.0] * count
            fresh = [True] * count  # No decision taken yet in the current episode.
            epsilon = config.epsilon(step)

            def write_episode(index, ended, truncated):
                result = EpisodeResult.from_info(episode_seeds[index], totals[index], infos[index],
                                                 truncated, budget_end=not ended)
                episode_writer.writerow({'episode': episode_ids[index], 'global_step': step,
                                         'epsilon': epsilon, **result.to_dict(), 'env': index})
                episodes_file.flush()

            while step < config.steps:
                # The last round may need fewer decisions than environments.
                active = min(count, config.steps - step)
                epsilon = config.epsilon(step)
                actions = agent.select_actions(np.stack(observations[:active]), epsilon)
                results = environments.step(actions)  # All emulators advance together.
                for index, (action, result) in enumerate(zip(actions, results)):
                    next_observation, reward, terminated, truncated, info = result
                    replay.add(observations[index], action, reward, next_observation,
                               terminated, truncated)
                    observations[index], infos[index] = next_observation, info
                    totals[index] += reward
                    fresh[index] = False
                    step += 1  # Checkpoints count completed decision cycles.
                    # Replay is new on resume: refill it without resetting epsilon's
                    # global schedule or learning from an empty/stale buffer.
                    if (step - start_step >= config.warmup and len(replay) >= config.batch_size
                            and step % config.learn_every == 0):
                        latest_loss = agent.learn(replay)
                        update_writer.writerow({'step': step, 'update': agent.updates,
                                                'loss': latest_loss, 'epsilon': epsilon})
                    ended = terminated or truncated
                    if index == 0:
                        # Count decisions, never raw emulator frames. Show the first
                        # emulator's episode before reset clears its reward/position.
                        progress.set_postfix(episodio=episodes_started, updates=agent.updates,
                                             epsilon=f'{epsilon:.3f}',
                                             loss='-' if latest_loss is None else f'{latest_loss:.4f}',
                                             reward=f'{totals[0]:.1f}', x=info['progress_max'],
                                             refresh=False)
                    progress.update(1)
                    if ended:
                        write_episode(index, ended=True, truncated=truncated)
                        episode += 1
                        episode_ids[index] = episodes_started
                        episode_seeds[index] = config.seed + episodes_started
                        episodes_started += 1
                        observations[index], infos[index] = environments.reset(
                            index, seed=episode_seeds[index])
                        totals[index] = 0.0
                        fresh[index] = True
                    if step == config.steps:
                        # Budget exhausted: record every unfinished episode once.
                        for other in range(count):
                            if not fresh[other]:
                                write_episode(other, ended=False, truncated=False)
                    if step % config.checkpoint_every == 0:
                        save_checkpoint(f'step-{step:09d}.pt')
                    if step % config.eval_every == 0 or step == config.steps:
                        # No epsilon/RNG consumption or replay changes in evaluation.
                        policy = NetworkPolicy(agent, f'DQN · decisión {step}')
                        progress.set_description('Evaluando')
                        try:
                            evaluation = evaluate(policy, environment_config, config.eval_episodes,
                                                  seed=config.seed + 10_000)
                        finally:
                            agent.online.train()
                            progress.set_description('Entrenando')
                        row = {key: evaluation[key] for key in ('success_rate', 'mean_reward', 'mean_progress')}
                        evaluation_writer.writerow({'step': step, **row})
                        evaluation_file.flush()
                        updates_file.flush()
                        score = (row['success_rate'], row['mean_progress'])
                        if score > best_score:
                            best_score = score
                            save_checkpoint('best.pt')
                        tqdm.write(f'Evaluación: paso={step:,} updates={agent.updates} epsilon={epsilon:.3f} '
                                   f'éxito={row["success_rate"]:.1%} x={row["mean_progress"]:.1f}',
                                   file=sys.stderr)
            save_checkpoint('final.pt')
            summary = {'steps': step, 'updates': agent.updates, 'episodes_finished': episode,
                       'steps_executed': step - start_step, 'resume': resume_info,
                       'finite_losses': True, 'last_loss': latest_loss,
                       'environment': identity, 'best_success_rate': best_score[0],
                       'checkpoint': str((output / 'final.pt').resolve()),
                       'warning': 'Synthetic fixture; NOT Mario or NES' if environment_config.demo else None}
            write_json(output / 'summary.json', summary)
            return summary
    except KeyboardInterrupt:
        if output_ready:
            save_checkpoint('interrupted.pt')
        raise
    finally:
        environments.close()

"""Readable training loop: interact → remember → learn → evaluate → checkpoint."""
from dataclasses import asdict
import csv
from pathlib import Path
import sys

from tqdm import tqdm

from .agent import DQNAgent
from .config import EnvironmentConfig, TrainConfig
from .environment import make_environment
from .evaluation import evaluate
from .metrics import EpisodeResult, runtime_metadata, write_json
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
    environment = make_environment(environment_config)
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
        identity = environment.identity()
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
        save_checkpoint(f'step-{step:09d}.pt')
        with tqdm(total=config.steps, initial=start_step, desc='Entrenando', unit='decisión',
                  dynamic_ncols=True, mininterval=0.5,
                  disable=not show_progress, file=sys.stderr) as progress, \
             (output / 'episodes.csv').open('w', newline='') as episodes_file, \
             (output / 'updates.csv').open('w', newline='') as updates_file, \
             (output / 'evaluation.csv').open('w', newline='') as evaluation_file:
            episode_writer = csv.DictWriter(episodes_file, fieldnames=[
                'episode', 'global_step', 'epsilon', 'seed', 'reward', 'progress_max',
                'decisions', 'raw_frames', 'success', 'death', 'truncated', 'budget_end'])
            update_writer = csv.DictWriter(updates_file, fieldnames=['step', 'update', 'loss', 'epsilon'])
            evaluation_writer = csv.DictWriter(evaluation_file, fieldnames=[
                'step', 'success_rate', 'mean_reward', 'mean_progress'])
            for writer in (episode_writer, update_writer, evaluation_writer):
                writer.writeheader()
            episode_seed = config.seed
            observation, info = environment.reset(seed=episode_seed)
            total_reward = 0.0
            for next_step in range(start_step + 1, config.steps + 1):
                epsilon = config.epsilon(next_step - 1)
                action = agent.select_action(observation, epsilon)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                replay.add(observation, action, reward, next_observation, terminated, truncated)
                observation = next_observation
                total_reward += reward
                # Replay is new on resume: refill it without resetting epsilon's
                # global schedule or learning from an empty/stale buffer.
                if (next_step - start_step >= config.warmup and len(replay) >= config.batch_size
                        and next_step % config.learn_every == 0):
                    latest_loss = agent.learn(replay)
                    update_writer.writerow({'step': next_step, 'update': agent.updates,
                                            'loss': latest_loss, 'epsilon': epsilon})
                step = next_step  # Checkpoints count completed decision cycles.
                ended = terminated or truncated
                # Count completed decisions, never raw emulator frames. Report the
                # current episode before reset clears its reward and position.
                progress.set_postfix(episodio=episode + 1, updates=agent.updates,
                                     epsilon=f'{epsilon:.3f}',
                                     loss='-' if latest_loss is None else f'{latest_loss:.4f}',
                                     reward=f'{total_reward:.1f}', x=info['progress_max'],
                                     refresh=False)
                progress.update(1)
                if ended or step == config.steps:
                    result = EpisodeResult.from_info(episode_seed, total_reward, info,
                                                      truncated, budget_end=not ended)
                    episode_writer.writerow({'episode': episode, 'global_step': step,
                                             'epsilon': epsilon, **result.to_dict()})
                    episodes_file.flush()
                    if ended:
                        episode += 1
                        episode_seed = config.seed + episode
                        observation, info = environment.reset(seed=episode_seed)
                        total_reward = 0.0
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
        environment.close()

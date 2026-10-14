"""Evaluation uses separate environments, no replay, optimizer or exploration."""
from .environment import make_environment
from .metrics import EpisodeResult


def evaluate(policy, config, episodes: int, seed: int) -> dict:
    environment = make_environment(config)
    results = []
    try:
        identity = environment.identity()
        for episode in range(episodes):
            episode_seed = seed + episode
            observation, info = environment.reset(seed=episode_seed)
            total = 0.0
            ended = False
            while not ended:
                action = policy.select_action(observation)
                observation, reward, terminated, truncated, info = environment.step(action)
                total += reward
                ended = terminated or truncated
            results.append(EpisodeResult.from_info(episode_seed, total, info, truncated).to_dict())
    finally:
        environment.close()
    return {'environment': identity, 'policy': policy.label, 'episodes': results,
            'success_rate': sum(row['success'] for row in results) / len(results),
            'mean_reward': sum(row['reward'] for row in results) / len(results),
            'mean_progress': sum(row['progress_max'] for row in results) / len(results)}

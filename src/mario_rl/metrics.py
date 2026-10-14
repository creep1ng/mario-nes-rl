"""Plain JSON/CSV artifacts make runs inspectable without a tracking service."""
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import platform

import torch


@dataclass
class EpisodeResult:
    seed: int
    reward: float
    progress_max: int
    decisions: int
    raw_frames: int
    success: bool
    death: bool
    truncated: bool
    budget_end: bool = False

    @classmethod
    def from_info(cls, seed, reward, info, truncated, budget_end=False):
        return cls(seed, float(reward), int(info['progress_max']),
                   int(info['decisions']), int(info['raw_frames']),
                   bool(info['flag_get']), bool(info['death']), bool(truncated), budget_end)

    def to_dict(self):
        return asdict(self)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def runtime_metadata() -> dict:
    root = Path(__file__).parent
    files = sorted(path for path in root.rglob('*')
                   if path.suffix in {'.py', '.html', '.css', '.js'})
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    packages = {name: version(name) for name in ('mario-rl', 'torch', 'numpy', 'pillow', 'gymnasium', 'tqdm')}
    try:
        packages['nes-py'] = version('nes-py')
    except PackageNotFoundError:
        packages['nes-py'] = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'torch_threads': torch.get_num_threads(),
            'source_sha256': digest.hexdigest()}

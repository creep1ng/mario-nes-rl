"""Public CLI resume acceptance; failure modes/contract precede implementation.

Artifacts: complete commands, stdout/stderr, CSV, checkpoints and safe-load proofs.
Optional --rom and --checkpoint verify real NES and an existing user checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(left, right):
    """Inspect persisted CLI artifacts, including Adam moments, not just weights."""
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def load(path):
    return torch.load(path, weights_only=True, map_location='cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rom')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    evidence = ROOT / 'artifacts' / 'resume' / f'run-{time.time_ns()}'
    evidence.mkdir(parents=True)
    commands = []
    results = []

    def command(*arguments):
        cmd = [sys.executable, '-m', 'mario_rl', *map(str, arguments)]
        commands.append(cmd)
        return cmd

    def run(name, *arguments, success=True):
        result = subprocess.run(command(*arguments), cwd=ROOT, capture_output=True,
                                text=True, timeout=240)
        (evidence / f'{name}.stdout.log').write_text(result.stdout)
        (evidence / f'{name}.stderr.log').write_text(result.stderr)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        return (json.loads(result.stdout) if success else result.stderr), result.stderr

    cases = [('demo', ['--demo'])]
    if args.rom:
        cases.append(('nes', ['--rom', str(Path(args.rom).resolve())]))
    for mode, source in cases:
        base = evidence / f'{mode}-base'
        run(f'{mode}-base', 'train', *source, '--steps', '40', '--warmup', '8',
            '--batch-size', '4', '--replay-capacity', '128', '--epsilon-decay', '100',
            '--gamma', '0.97', '--seed', '11', '--target-every', '100000',
            '--action-repeat', '2', '--max-episode-steps', '16',
            '--checkpoint-every', '20', '--eval-every', '20', '--eval-episodes', '1',
            '--output', base)
        original_hashes = {p.name: sha(p) for p in base.iterdir() if p.is_file()}
        checkpoint = base / 'final.pt'
        original = load(checkpoint)
        output = evidence / f'{mode}-continued'
        summary, terminal = run(f'{mode}-continued', 'train', *source,
                                '--resume', checkpoint, '--steps', '56',
                                '--checkpoint-every', '8', '--eval-every', '8',
                                '--output', output)
        assert summary['steps'] == 56 and summary['steps_executed'] == 16
        assert summary['resume']['start_step'] == 40
        assert summary['resume']['sha256'] == sha(checkpoint)
        assert summary['resume']['replay_reset'] is True
        assert '40/56' in terminal and '56/56' in terminal and 'Reanudación' in terminal
        initial = load(output / 'step-000000040.pt')
        for key in ['online', 'target', 'optimizer', 'step', 'updates']:
            same(original[key], initial[key])
        inherited = json.loads((output / 'config.json').read_text())
        for key, value in original['config'].items():
            if key not in ['steps', 'eval_every', 'checkpoint_every']:
                assert inherited['training'][key] == value, key
        assert inherited['environment']['action_repeat'] == 2
        assert inherited['environment']['max_episode_steps'] == 16
        rows = list(csv.DictReader((output / 'updates.csv').open()))
        assert [int(row['step']) for row in rows] == [48, 52, 56]
        assert int(rows[0]['update']) == original['updates'] + 1
        expected_epsilon = 1 + ((48 - 1 - 8) / 100) * (.05 - 1)
        assert abs(float(rows[0]['epsilon']) - expected_epsilon) < 1e-12
        final = load(output / 'final.pt')
        assert final['updates'] == original['updates'] + 3
        assert any(not torch.equal(final['online'][k], original['online'][k]) for k in final['online'])
        for key in original['optimizer']['state']:
            assert final['optimizer']['state'][key]['step'] > original['optimizer']['state'][key]['step']
        run(f'{mode}-evaluate', 'evaluate', *source, '--checkpoint', output / 'final.pt',
            '--action-repeat', '2', '--max-episode-steps', '16', '--episodes', '1',
            '--output', evidence / f'{mode}-evaluation.json')
        for label, extra in [('budget', ['--steps', '40']),
                             ('repeat', ['--steps', '56', '--action-repeat', '3'])]:
            invalid_output = evidence / f'{mode}-invalid-{label}'
            run(f'{mode}-invalid-{label}', 'train', *source, '--resume', checkpoint,
                *extra, '--output', invalid_output, success=False)
            assert not invalid_output.exists()
        run(f'{mode}-protect-source', 'train', *source, '--resume', checkpoint,
            '--steps', '56', '--output', base, success=False)
        assert original_hashes == {p.name: sha(p) for p in base.iterdir() if p.is_file()}
        assert not list(output.glob('*.tmp'))
        results.append({'mode': mode, 'start_step': 40, 'final_step': 56,
                        'steps_executed': 16, 'restored': ['online', 'target', 'Adam', 'updates'],
                        'new_updates': 3, 'inherited_schedule': True, 'source_unchanged': True})

    corrupt = evidence / 'corrupt.pt'
    corrupt.write_text('not a checkpoint')
    for name, checkpoint in [('missing', evidence / 'missing.pt'), ('corrupt', corrupt)]:
        output = evidence / f'invalid-{name}'
        run(name, 'train', '--demo', '--resume', checkpoint, '--steps', '56',
            '--output', output, success=False)
        assert not output.exists()

    # Interrupt our own resumed CLI, then reload its saved checkpoint in a fresh process.
    interrupted = evidence / 'interrupted'
    source = evidence / 'demo-continued' / 'final.pt'
    stdout_path, stderr_path = evidence / 'interrupt.stdout.log', evidence / 'interrupt.stderr.log'
    with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
        process = subprocess.Popen(command('train', '--demo', '--resume', source,
                                   '--steps', '100000', '--output', interrupted),
                                   cwd=ROOT, stdout=stdout, stderr=stderr)
        try:
            deadline = time.monotonic() + 90
            while 'step-000000064.pt' not in stderr_path.read_text():
                assert process.poll() is None, stderr_path.read_text()
                assert time.monotonic() < deadline
                time.sleep(.05)
            process.send_signal(signal.SIGINT)
            process.wait(timeout=30)
            assert process.returncode == 130
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    stopped = load(interrupted / 'interrupted.pt')
    assert stopped['step'] >= 64 and stopped['updates'] > load(source)['updates']
    assert '100%' not in stderr_path.read_text()
    run('interrupt-evaluation', 'evaluate', '--demo', '--action-repeat', '2',
        '--checkpoint', interrupted / 'interrupted.pt', '--episodes', '1',
        '--max-episode-steps', '16', '--output', evidence / 'interrupt-evaluation.json')

    if args.checkpoint:
        assert args.rom, '--checkpoint requires --rom'
        path = args.checkpoint.resolve()
        before, old = sha(path), load(path)
        output = evidence / 'user-checkpoint-continued'
        summary, _ = run('user-checkpoint', 'train', '--rom', args.rom, '--resume', path,
                          '--steps', old['step'] + 40, '--output', output)
        initial = load(output / f'step-{old["step"]:09d}.pt')
        for key in ['online', 'target', 'optimizer', 'step', 'updates']:
            same(old[key], initial[key])
        assert sha(path) == before and summary['steps_executed'] == 40
        results.append({'mode': 'user NES checkpoint', 'path': str(path),
                        'start_step': old['step'], 'steps_executed': 40,
                        'source_unchanged': True, 'restored': ['online', 'target', 'Adam', 'updates']})

    from mario_rl.metrics import runtime_metadata
    report = {'status': 'passed', 'command': [sys.executable, __file__, *sys.argv[1:]],
              'runtime': runtime_metadata(), 'results': results, 'commands': commands,
              'evidence_directory': str(evidence), 'resume_interrupt_step': stopped['step']}
    (evidence / 'report.json').write_text(json.dumps(report, indent=2))
    (evidence.parent / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

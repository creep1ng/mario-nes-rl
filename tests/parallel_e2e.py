"""CLI acceptance for parallel emulators (--envs). Same style: real processes, no mocks.

Run: .venv/bin/python tests/parallel_e2e.py [--rom /absolute/game.nes] [--device cuda]
Checks exact decision accounting, CSV/checkpoint contracts, Ctrl+C on the whole
process group (as a terminal sends it), no orphan workers, and resume with --envs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]


def group_alive(pgid: int) -> list[str]:
    found = subprocess.run(['pgrep', '-g', str(pgid)], capture_output=True, text=True)
    return found.stdout.split()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rom')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    evidence = ROOT / 'artifacts' / 'parallel' / f'run-{time.time_ns()}'
    evidence.mkdir(parents=True)
    cases = [('demo', ['--demo'])]
    if args.rom:
        cases.append(('nes', ['--rom', str(Path(args.rom).resolve())]))
    results = []

    def cli(*arguments):
        return [sys.executable, '-m', 'mario_rl', *map(str, arguments)]

    def run(name, *arguments):
        done = subprocess.run(cli(*arguments), cwd=ROOT, capture_output=True, text=True, timeout=300)
        (evidence / f'{name}.stdout.json').write_text(done.stdout)
        (evidence / f'{name}.stderr.log').write_text(done.stderr)
        assert done.returncode == 0, done.stdout + done.stderr
        return json.loads(done.stdout), done.stderr

    for mode, source in cases:
        # 50 is not a multiple of 4: the last round must stop at exactly 50.
        common = ['train', *source, '--steps', '50', '--warmup', '8', '--batch-size', '4',
                  '--replay-capacity', '128', '--max-episode-steps', '16',
                  '--eval-every', '25', '--eval-episodes', '1', '--checkpoint-every', '10',
                  '--seed', '7', '--device', args.device]
        output = evidence / f'{mode}-envs4'
        summary, terminal = run(f'{mode}-envs4', *common, '--envs', '4', '--output', output)
        assert summary['steps'] == 50 and summary['updates'] > 0, summary
        assert 'Emuladores paralelos: 4 procesos' in terminal
        for step in (0, 10, 20, 30, 40, 50):
            assert (output / f'step-{step:09d}.pt').exists(), step
        assert json.loads((output / 'config.json').read_text())['training']['envs'] == 4
        rows = list(csv.DictReader((output / 'episodes.csv').open()))
        assert {row['env'] for row in rows} == {'0', '1', '2', '3'}
        assert len({row['episode'] for row in rows}) == len(rows), 'episode ids must be unique'
        assert all(int(row['seed']) == 7 + int(row['episode']) for row in rows)
        updates = list(csv.DictReader((output / 'updates.csv').open()))
        assert all(int(row['step']) % 4 == 0 for row in updates)
        assert len(updates) == summary['updates']

        # Determinism: the same parallel run twice gives the same learning. Only
        # on CPU: cuDNN convolution gradients use atomics and differ in last bits.
        if args.device == 'cpu':
            again = evidence / f'{mode}-envs4-again'
            run(f'{mode}-envs4-again', *common, '--envs', '4', '--output', again)
            for name in ('episodes.csv', 'updates.csv', 'evaluation.csv'):
                assert (again / name).read_text() == (output / name).read_text(), name

        # Resume a single-emulator checkpoint with more emulators.
        single = evidence / f'{mode}-envs1'
        run(f'{mode}-envs1', *common, '--output', single)
        resumed = evidence / f'{mode}-resume-envs3'
        resumed_summary, _ = run(f'{mode}-resume-envs3', 'train', *source, '--resume',
                                 single / 'final.pt', '--envs', '3', '--steps', '80',
                                 '--checkpoint-every', '10', '--eval-every', '30',
                                 '--device', args.device, '--output', resumed)
        assert resumed_summary['steps'] == 80 and resumed_summary['steps_executed'] == 30
        before = torch.load(single / 'final.pt', weights_only=True, map_location='cpu')
        start = torch.load(resumed / 'step-000000050.pt', weights_only=True, map_location='cpu')
        assert all(torch.equal(before['online'][k], start['online'][k]) for k in before['online'])

        # Ctrl+C from a terminal reaches every process of the foreground group.
        interrupted = evidence / f'{mode}-interrupted'
        stderr_path = evidence / f'{mode}-interrupted.stderr.log'
        with stderr_path.open('w') as stderr:
            process = subprocess.Popen(cli('train', *source, '--envs', '4', '--steps', '100000',
                                           '--warmup', '8', '--batch-size', '4',
                                           '--replay-capacity', '128', '--max-episode-steps', '16',
                                           '--checkpoint-every', '16', '--device', args.device,
                                           '--output', interrupted),
                                       cwd=ROOT, stdout=subprocess.DEVNULL, stderr=stderr,
                                       start_new_session=True)
            try:
                deadline = time.monotonic() + 120
                while 'step-000000032.pt' not in stderr_path.read_text():
                    assert process.poll() is None, stderr_path.read_text()
                    assert time.monotonic() < deadline, 'Checkpoint was not announced'
                    time.sleep(.05)
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=60)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        assert process.returncode == 130, stderr_path.read_text()
        assert (interrupted / 'interrupted.pt').exists()
        time.sleep(.5)
        assert not group_alive(process.pid), 'orphan emulator workers remain'
        assert not list(interrupted.glob('*.tmp'))
        results.append({'mode': mode, 'device': args.device, 'envs': 4,
                        'exact_steps': True, 'deterministic': args.device == 'cpu', 'resume_envs3': True,
                        'interrupt_exit': process.returncode, 'orphans': 0})

    report = {'status': 'passed', 'command': [sys.executable, __file__, *sys.argv[1:]],
              'results': results, 'evidence_directory': str(evidence)}
    (evidence / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

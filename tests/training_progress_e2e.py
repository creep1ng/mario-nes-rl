"""CLI acceptance for progress/checkpoints; contract written before implementation.

Run with optional --rom /absolute/game.nes. No mocks or isolated unit tests.
Artifacts include commands, raw terminal output, load evaluations and a report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rom')
    args = parser.parse_args()
    evidence = ROOT / 'artifacts' / 'training-progress' / f'run-{time.time_ns()}'
    evidence.mkdir(parents=True)
    cases = [('demo', ['--demo'])]
    if args.rom:
        cases.append(('nes', ['--rom', str(Path(args.rom).resolve())]))
    commands = []
    results = []

    def cli(*arguments):
        command = [sys.executable, '-m', 'mario_rl', *map(str, arguments)]
        commands.append(command)
        return command

    def run(name, *arguments):
        completed = subprocess.run(cli(*arguments), cwd=ROOT, capture_output=True,
                                   text=True, timeout=180)
        (evidence / f'{name}.stdout.json').write_text(completed.stdout)
        (evidence / f'{name}.stderr.log').write_text(completed.stderr)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        # JSON remains machine-readable without stripping progress/log lines.
        return json.loads(completed.stdout), completed.stderr

    for mode, source in cases:
        common = ['train', *source, '--steps', '48', '--warmup', '8',
                  '--batch-size', '4', '--replay-capacity', '128',
                  '--max-episode-steps', '16', '--eval-every', '24',
                  '--eval-episodes', '1', '--checkpoint-every', '16', '--seed', '7']
        output = evidence / f'{mode}-progress'
        summary, terminal = run(f'{mode}-progress', *common, '--output', output)
        assert summary['steps'] == 48 and summary['updates'] > 0
        assert '100%' in terminal and '48/48' in terminal
        assert all(label in terminal for label in ['epsilon=', 'loss=', 'episodio=',
                                                   'updates=', 'reward=', 'x='])
        assert 'Evaluando' in terminal and 'Checkpoint:' in terminal
        snapshots = ['step-000000000.pt', 'step-000000016.pt',
                     'step-000000032.pt', 'step-000000048.pt', 'best.pt', 'final.pt']
        for index, name in enumerate(snapshots):
            checkpoint = output / name
            assert checkpoint.exists(), name
            run(f'{mode}-reload-{index}', 'evaluate', *source, '--checkpoint', checkpoint,
                '--episodes', '1', '--max-episode-steps', '16',
                '--output', evidence / f'{mode}-reload-{index}.json')
        assert not list(output.glob('*.tmp'))

        quiet = evidence / f'{mode}-no-progress'
        quiet_summary, quiet_terminal = run(f'{mode}-no-progress', *common,
                                            '--no-progress', '--output', quiet)
        assert '100%' not in quiet_terminal and '48/48' not in quiet_terminal
        assert quiet_summary['updates'] == summary['updates']
        for name in ['episodes.csv', 'updates.csv', 'evaluation.csv']:
            assert (quiet / name).read_text() == (output / name).read_text(), name

        interrupted = evidence / f'{mode}-interrupted'
        stdout_path = evidence / f'{mode}-interrupted.stdout.log'
        stderr_path = evidence / f'{mode}-interrupted.stderr.log'
        interrupt_command = cli('train', *source, '--steps', '100000', '--warmup', '8',
                                '--batch-size', '4', '--replay-capacity', '128',
                                '--max-episode-steps', '16', '--checkpoint-every', '16',
                                '--output', interrupted)
        with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
            process = subprocess.Popen(interrupt_command, cwd=ROOT, stdout=stdout, stderr=stderr)
            try:
                deadline = time.monotonic() + 90
                while 'step-000000016.pt' not in stderr_path.read_text():
                    assert process.poll() is None, stderr_path.read_text()
                    assert time.monotonic() < deadline, 'Checkpoint was not announced'
                    time.sleep(.05)
                process.send_signal(signal.SIGINT)  # Only our own CLI child.
                process.wait(timeout=30)
                assert process.returncode == 130, stderr_path.read_text()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        assert (interrupted / 'interrupted.pt').exists()
        assert '100%' not in stderr_path.read_text()
        assert 'interrupted.pt' in stderr_path.read_text()
        assert not list(interrupted.glob('*.tmp'))
        run(f'{mode}-reload-interrupted', 'evaluate', *source,
            '--checkpoint', interrupted / 'interrupted.pt', '--episodes', '1',
            '--max-episode-steps', '16', '--output', evidence / f'{mode}-interrupt-evaluation.json')
        results.append({'mode': mode, 'steps': summary['steps'], 'updates': summary['updates'],
                        'checkpoints_loaded': snapshots + ['interrupted.pt'],
                        'no_progress_same_learning': True, 'interrupt_exit': process.returncode})

    from mario_rl.metrics import runtime_metadata
    report = {'status': 'passed', 'command': [sys.executable, __file__, *sys.argv[1:]],
              'runtime': runtime_metadata(), 'results': results, 'commands': commands,
              'evidence_directory': str(evidence)}
    (evidence / 'report.json').write_text(json.dumps(report, indent=2))
    (evidence.parent / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""Public CLI/HTTP acceptance harness. Initial contract preceded application code.

Run: .venv/bin/python tests/e2e.py [--rom /absolute/path/to/game.nes]
No unit tests. Each invocation preserves a separate evidence directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'e2e'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args: str, expect_success: bool = True):
    result = subprocess.run([sys.executable, '-m', 'mario_rl', *args], cwd=ROOT,
                            text=True, capture_output=True, timeout=180)
    with (OUT / 'commands.log').open('a') as log:
        log.write(f'$ {sys.executable} -m mario_rl {" ".join(args)}\n')
        log.write(result.stdout + result.stderr + f'\nexit={result.returncode}\n')
    assert (result.returncode == 0) == expect_success, result.stdout + result.stderr
    return result


def request(base: str, path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(base + path, data=body, headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def check_viewer(source: list[str], checkpoints: list[Path], mode: str, limit: int) -> dict:
    """Exercise one complete comparison through HTTP, including its MP4 artifact."""
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    hashes_before = [digest(path) for path in checkpoints]
    log_path = OUT / f'{mode}-server.log'
    log = log_path.open('w')
    args = [sys.executable, '-m', 'mario_rl', 'compare', *source,
            '--port', str(port), '--max-episode-steps', str(limit),
            '--output', str(OUT / f'{mode}-viewer')]
    for path in checkpoints:
        args += ['--checkpoint', str(path)]
    server = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=log)
    base = f'http://127.0.0.1:{port}'
    try:
        for _ in range(150):
            try:
                state = request(base, '/api/state')
                break
            except (URLError, ConnectionError):
                if server.poll() is not None:
                    raise AssertionError(log_path.read_text())
                time.sleep(.1)
        else:
            raise AssertionError('Viewer did not start')
        assert state['mode'] == mode and len(state['panels']) == 3
        assert all(p['steps'] == 0 for p in state['panels'])
        assert all(p['frame'].startswith('data:image/png;base64,') for p in state['panels'])
        assert request(base, '/api/state')['tick'] == 0  # Paused: GET has no side effects.
        moved = request(base, '/api/step', {})
        assert moved['tick'] == 1 and all(p['steps'] == 1 for p in moved['panels'])
        selected = request(base, '/api/select', {'panel': 1, 'policy': 'checkpoint:0'})
        assert selected['tick'] == 0 and selected['panels'][1]['policy'] == 'checkpoint:0'
        request(base, '/api/select', {'panel': 1, 'policy': 'right'})
        request(base, '/api/record/start', {'fps': 15})
        for _ in range(limit + 4):
            final = request(base, '/api/step', {})
        recording = request(base, '/api/record/stop', {})
        video = Path(recording['video'])
        assert video.exists() and video.stat().st_size > 1000
        probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                                'stream=codec_name,width,height,nb_frames',
                                '-of', 'json', str(video)], capture_output=True, text=True, check=True)
        assert request(base, '/api/state')['tick'] == final['tick']
        assert all(p['done'] for p in final['panels'])
        reset = request(base, '/api/reset', {})
        assert reset['tick'] == 0 and all(not p['done'] for p in reset['panels'])
        exported = request(base, '/api/report')
        assert exported['mode'] == mode
        try:
            request(base, '/api/select', {'panel': 1, 'policy': '../../untrusted.pt'})
            raise AssertionError('Unknown policy accepted')
        except HTTPError as error:
            assert error.code == 400
        for path in ['/', '/app.js', '/style.css']:
            with urlopen(base + path) as response:
                assert response.status == 200 and len(response.read()) > 100
        # Cross-origin websites must not control the loopback emulator server.
        hostile = Request(base + '/api/step', data=b'{}',
                          headers={'Origin': 'https://untrusted.example'})
        try:
            urlopen(hostile)
            raise AssertionError('Cross-origin control accepted')
        except HTTPError as error:
            assert error.code == 403
        assert hashes_before == [digest(path) for path in checkpoints]
        result = {'mode': mode, 'video': str(video), 'sha256': digest(video),
                  'probe': json.loads(probe.stdout), 'final_panels': [
                      {k: v for k, v in p.items() if k != 'frame'} for p in final['panels']],
                  'checkpoints_unchanged': True}
        (OUT / f'{mode}-viewer-result.json').write_text(json.dumps(result, indent=2))
        return result
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log.close()


def check_training(source: list[str], mode: str, steps: int, limit: int):
    output = OUT / f'{mode}-training'
    command('train', *source, '--steps', str(steps), '--warmup', '8',
            '--batch-size', '4', '--replay-capacity', '128',
            '--max-episode-steps', str(limit), '--eval-every', str(steps // 2),
            '--eval-episodes', '2', '--checkpoint-every', str(steps // 2),
            '--target-every', '2', '--output', str(output), '--seed', '7')
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['updates'] > 0 and summary['finite_losses'] and summary['steps'] == steps
    assert (output / 'episodes.csv').stat().st_size > 50
    assert (output / 'evaluation.csv').stat().st_size > 50
    checkpoint = output / 'final.pt'
    assert checkpoint.exists()
    for name in ['first', 'second']:
        command('evaluate', *source, '--checkpoint', str(checkpoint), '--episodes', '2',
                '--max-episode-steps', str(limit), '--seed', '19',
                '--output', str(OUT / f'{mode}-evaluation-{name}.json'))
    first = json.loads((OUT / f'{mode}-evaluation-first.json').read_text())
    second = json.loads((OUT / f'{mode}-evaluation-second.json').read_text())
    assert first['episodes'] == second['episodes']
    assert digest(checkpoint) != digest(output / 'step-000000000.pt')
    return summary, checkpoint


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--rom')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    evidence_root = args.output.resolve()
    OUT = evidence_root / f'run-{time.time_ns()}'
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'python': sys.version, 'command': f'{sys.executable} tests/e2e.py',
              'evidence_directory': str(OUT), 'checks': [],
              'real_mario': 'not run: no ROM supplied'}
    missing = command('evaluate', '--rom', str(OUT / 'missing.nes'),
                      '--episodes', '1', expect_success=False)
    assert 'ROM' in missing.stderr
    corrupt = OUT / 'corrupt.pt'
    corrupt.write_text('not a PyTorch checkpoint')
    command('evaluate', '--demo', '--checkpoint', str(corrupt), expect_success=False)
    report['checks'].append('Missing ROM and corrupt checkpoint fail clearly, without fallback')

    report['synthetic_training'], checkpoint = check_training(['--demo'], 'demo', 48, 8)
    command('smoke', '--demo', '--output', str(OUT / 'synthetic-smoke'))
    smoke = json.loads((OUT / 'synthetic-smoke' / 'report.json').read_text())
    assert smoke['success_seen'] and smoke['death_seen'] and smoke['truncation_seen']
    assert smoke['frames_are_independent'] and smoke['repeat_stops_at_end']
    report['checks'].append('Synthetic smoke covers success, death, truncation and immutable frames')
    report['synthetic_viewer'] = check_viewer(['--demo'], [checkpoint], 'demo', 8)

    if args.rom:
        source = ['--rom', str(Path(args.rom).resolve())]
        command('smoke', *source, '--output', str(OUT / 'real-smoke'))
        report['real_mario'] = json.loads((OUT / 'real-smoke' / 'report.json').read_text())
        assert report['real_mario']['real_mario'] and report['real_mario']['death_seen']
        report['nes_training'], native_checkpoint = check_training(source, 'nes', 256, 128)
        report['nes_viewer'] = check_viewer(source, [native_checkpoint], 'nes', 128)
        mismatch = command('evaluate', '--demo', '--checkpoint', str(native_checkpoint), expect_success=False)
        assert 'incompatible' in mismatch.stderr
        report['checks'].append('Real NES: boot, movement, death, training, reload, evaluation, comparison and MP4')
    report['checks'] += ['Finite learning updates + reproducible checkpoint evaluation in fresh processes',
                         'HTTP: pause/read, step, select, reset, terminal freeze, JSON export, MP4 and origin guard',
                         'Frozen evaluation leaves checkpoint files unchanged']
    report['status'] = 'passed: synthetic and real NES E2E' if args.rom else 'passed: synthetic E2E only'
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    (evidence_root / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

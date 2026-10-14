"""Record the actual comparison frames, including labels and the fixture warning."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw


class VideoRecorder:
    def __init__(self, directory: Path, width: int, height: int, fps: int):
        executable = shutil.which('ffmpeg')
        if not executable:
            raise ValueError('Grabar MP4 requiere ffmpeg instalado en el sistema.')
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f'comparison-{time.time_ns()}.mp4'
        self.log_path = self.path.with_suffix('.ffmpeg.log')
        self.log = self.log_path.open('w')
        self.frames = 0
        self.fps = fps
        self.process = subprocess.Popen([
            executable, '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
            '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
            '-preset', 'ultrafast', '-crf', '25', '-pix_fmt', 'yuv420p', str(self.path),
        ], stdin=subprocess.PIPE, stderr=self.log)

    def append(self, image: Image.Image):
        try:
            self.process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
            self.frames += 1
        except BrokenPipeError as error:
            raise ValueError(f'ffmpeg falló; consulta {self.log_path}') from error

    def close(self):
        self.process.stdin.close()
        try:
            code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise ValueError('ffmpeg no terminó la grabación en 30 segundos.')
        finally:
            self.log.close()
        if code != 0:
            raise ValueError(f'ffmpeg terminó con código {code}; consulta {self.log_path}')
        return {'video': str(self.path.resolve()), 'frames': self.frames,
                'fps': self.fps, 'download': '/api/video'}


def mosaic(panels, demo: bool, tick: int) -> Image.Image:
    """A frame per agent decision, not an audio-enabled emulator capture."""
    width = 512 * len(panels)
    image = Image.new('RGB', (width, 584), '#11192a')
    draw = ImageDraw.Draw(image)
    title = 'SYNTHETIC FIXTURE - NOT MARIO / NOT NES' if demo else 'SUPER MARIO BROS - NES - LEVEL 1-1'
    draw.text((12, 12), f'{title} | decision {tick}', fill='#ffd585' if demo else '#a895ff')
    for index, panel in enumerate(panels):
        x = index * 512
        frame = Image.fromarray(panel.environment.render()).resize((512, 480), Image.Resampling.NEAREST)
        image.paste(frame, (x, 40))
        label = panel.policy.label.encode('ascii', 'replace').decode()
        draw.text((x + 12, 530), label, fill='#ffffff')
        draw.text((x + 12, 548), f'x={panel.info["progress_max"]} return={panel.reward:.2f} '
                  f'steps={panel.steps} done={panel.done}', fill='#c7d4e8')
    return image

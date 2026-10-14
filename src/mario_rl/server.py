"""Small loopback-only HTTP server. Static whitelist, frozen models, serialized ticks."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlparse


STATIC = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8')}
WEB = Path(__file__).parent / 'web'


def serve(runner, port: int):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type='application/json; charset=utf-8', download=None):
            if isinstance(body, dict):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; "
                             "script-src 'self'; style-src 'self'; frame-ancestors 'none'")
            if download:
                self.send_header('Content-Disposition', f'attachment; filename="{download}"')
            self.end_headers()
            self.wfile.write(body)

        def _local_request(self):
            # Guard Host/Origin against DNS rebinding and cross-site browser POSTs.
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            host = self.headers.get('Host', '')
            origin = self.headers.get('Origin')
            if host not in allowed or (origin and origin not in {f'http://{name}' for name in allowed}):
                self.send(403, {'error': 'Solo se permiten peticiones locales del visor.'})
                return False
            return True

        def do_GET(self):
            if not self._local_request():
                return
            route = urlparse(self.path).path
            if route in STATIC:
                filename, content_type = STATIC[route]
                self.send(200, (WEB / filename).read_bytes(), content_type)
            elif route == '/api/state':
                with lock:
                    self.send(200, runner.snapshot())
            elif route == '/api/report':
                with lock:
                    self.send(200, runner.report(), download='comparison.json')
            elif route == '/api/video':
                with lock:
                    if not runner.last_recording:
                        self.send(404, {'error': 'Aún no hay un vídeo finalizado.'})
                        return
                    path = Path(runner.last_recording['video'])
                    self.send(200, path.read_bytes(), 'video/mp4', path.name)
            else:
                self.send(404, {'error': 'Ruta desconocida.'})

        def do_POST(self):
            if not self._local_request():
                return
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 <= size <= 4096:
                    raise ValueError('Cuerpo de petición demasiado grande.')
                payload = json.loads(self.rfile.read(size) or b'{}')
                if not isinstance(payload, dict):
                    raise ValueError('Se espera un objeto JSON.')
                route = urlparse(self.path).path
                with lock:
                    if route == '/api/step':
                        result = runner.step()
                    elif route == '/api/reset':
                        result = runner.reset(payload.get('seed'))
                    elif route == '/api/select':
                        result = runner.select(payload.get('panel'), payload.get('policy'))
                    elif route == '/api/record/start':
                        result = runner.start_recording(payload.get('fps', 15))
                    elif route == '/api/record/stop':
                        result = runner.stop_recording()
                    else:
                        self.send(404, {'error': 'Ruta desconocida.'})
                        return
                self.send(200, result)
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.send(400, {'error': str(error)})
            except RuntimeError as error:
                self.send(500, {'error': str(error)})

        def log_message(self, format, *args):
            # Ticks are intentionally not noisy; failures still appear in browser.
            if len(args) > 1 and str(args[1]) != '200':
                super().log_message(format, *args)

    try:
        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    except OSError:
        runner.close()
        raise
    server.daemon_threads = True
    print(f'Visor: http://127.0.0.1:{server.server_port}', flush=True)
    print('DEMO SINTÉTICA: NO ES MARIO NI NES' if runner.config.demo else 'NES real: Super Mario Bros. 1-1', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        with lock:
            runner.close()

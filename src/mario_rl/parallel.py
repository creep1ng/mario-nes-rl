"""Several independent emulators, stepped together; one process per emulator.

The training loop always sees the same small interface, whatever the count:
``reset(index, seed)``, ``step(actions)``, ``identity()`` and ``close()``.

* One environment runs inside the training process: exactly the original loop.
* N > 1 environments run in N worker processes. ``step`` sends every action
  first and only then waits for the answers, so the emulators advance at the
  same time on different CPU cores while the agent stays in the main process.
"""
from __future__ import annotations

import multiprocessing
import signal

from .config import EnvironmentConfig
from .environment import make_environment


def _worker(connection, environment_config: EnvironmentConfig) -> None:
    """Own one emulator and obey commands until told to close."""
    # Ctrl+C reaches the whole process group; the parent alone decides when to
    # stop, saves interrupted.pt and then closes its workers in order.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    environment = None
    try:
        environment = make_environment(environment_config)
        connection.send(('ok', environment.identity()))
        while True:
            command, argument = connection.recv()
            if command == 'step':
                connection.send(('ok', environment.step(argument)))
            elif command == 'reset':
                connection.send(('ok', environment.reset(seed=argument)))
            elif command == 'close':
                break
    except (EOFError, BrokenPipeError):
        pass  # The parent disappeared; nothing left to answer.
    except Exception as error:  # Report to the parent instead of dying silently.
        try:
            connection.send(('error', f'{type(error).__name__}: {error}'))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if environment is not None:
            environment.close()
        connection.close()


class SerialEnvironments:
    """One environment in this process. No IPC, identical to the original loop."""

    def __init__(self, environment_config: EnvironmentConfig):
        self.environment = make_environment(environment_config)
        self.count = 1

    def identity(self) -> dict:
        return self.environment.identity()

    def reset(self, index: int, seed: int):
        return self.environment.reset(seed=seed)

    def step(self, actions: list[int]) -> list[tuple]:
        return [self.environment.step(action) for action in actions]

    def close(self) -> None:
        self.environment.close()


class ParallelEnvironments:
    """N emulators in N processes; the CPU cores do the emulation concurrently."""

    def __init__(self, environment_config: EnvironmentConfig, count: int):
        # "spawn" starts clean interpreters: no copied torch threads or locks.
        context = multiprocessing.get_context('spawn')
        self.count = count
        self.connections = []
        self.processes = []
        try:
            for _ in range(count):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, environment_config),
                                          daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            # Every emulator boots the ROM in parallel; wait for all of them.
            identities = [self._receive(index) for index in range(count)]
        except BaseException:
            self.close()
            raise
        if any(identity != identities[0] for identity in identities):
            self.close()
            raise ValueError('Los emuladores paralelos no tienen la misma identidad.')
        self._identity = identities[0]

    def _receive(self, index: int):
        try:
            status, payload = self.connections[index].recv()
        except EOFError as error:
            raise RuntimeError(f'El emulador paralelo {index} terminó inesperadamente.') from error
        if status == 'error':
            raise ValueError(f'Emulador paralelo {index}: {payload}')
        return payload

    def identity(self) -> dict:
        return self._identity

    def reset(self, index: int, seed: int):
        self.connections[index].send(('reset', seed))
        return self._receive(index)

    def step(self, actions: list[int]) -> list[tuple]:
        # Send everything first, then collect: this is where parallelism happens.
        for index, action in enumerate(actions):
            self.connections[index].send(('step', int(action)))
        return [self._receive(index) for index in range(len(actions))]

    def close(self) -> None:
        for connection in self.connections:
            try:
                connection.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        for connection in self.connections:
            connection.close()
        self.connections, self.processes = [], []


def make_environments(environment_config: EnvironmentConfig, count: int):
    if count < 1:
        raise ValueError('envs debe ser positivo.')
    if count == 1:
        return SerialEnvironments(environment_config)
    return ParallelEnvironments(environment_config, count)

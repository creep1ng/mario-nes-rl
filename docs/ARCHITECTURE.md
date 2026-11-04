# Arquitectura: DQN legible, evaluación independiente

## Decisiones principales

| Área | Decisión | Motivo |
|---|---|---|
| Juego | SMB1 NES estándar, nivel 1-1, una vida por intento. | Alcance verificable. |
| Emulador | `nes-py` con ROM externa. | No distribuir assets de Nintendo. |
| Algoritmo | DQN clásico: replay uniforme, target, epsilon-greedy. | Explicar cada paso sin framework RL. |
| UI | Navegador local, HTML/CSS/JS, stdlib HTTP. | No toolchain frontend ni backend complejo. |
| Concurrencia | Visor: tick serializado, un entorno por panel. Entrenamiento: `--envs N` procesos de emulador. | Aislamiento en el visor; throughput al entrenar. |
| Checkpoints | `weights_only=True`, compatibilidad explícita, escritura atómica. | Evitar pickle arbitrario y lecturas incompletas. |
| Fixture | Solo `--demo`, aviso visible. | Verificar infraestructura sin fingir NES. |

## Componentes

- `config.py`: dataclasses y validación de parámetros; todas las unidades explícitas.
- `environment.py`: ROM, emulador, arranque acotado, métricas RAM, recompensa y fin.
- `fixture.py`: plataforma geométrica sintética para E2E, no Nintendo.
- `preprocessing.py`: repetición de acción, límite externo y stack visual.
- `model.py`: tres convoluciones + FC512 + salida lineal Q de siete acciones.
- `replay.py`: memoria circular en arrays NumPy preasignados; muestreo uniforme
  idéntico al `deque` original, lote construido con un solo indexado por campo.
- `agent.py`: epsilon-greedy, Bellman target, Huber, Adam, target y checkpoints.
- `parallel.py`: N emuladores, uno por proceso `spawn`; `step` envía todas las
  acciones y luego recoge. Con N=1 el emulador vive en el proceso principal.
- `training.py`: coordinación explícita y artefactos CSV/JSON; cada ronda procesa
  las N transiciones como decisiones individuales, en orden de entorno.
- `evaluation.py`, `policies.py`: políticas congeladas, sin aprendizaje.
- `comparison.py`: paneles independientes, reset común y métricas.
- `server.py`, `web/`: HTTP local y controles; GET no avanza la simulación.
- `recording.py`: mosaico RGB real, MP4 y metadatos.
- `smoke.py`, `tests/e2e.py`: verificaciones de comportamiento y evidencia.

## Estado visual y acciones

Las acciones son: NOOP, derecha, derecha+A, derecha+B, derecha+A+B, A e izquierda.
Las máscaras NES corresponden a `0,128,129,130,131,1,64`.

Cada decisión repite una acción hasta cuatro frames, sumando recompensas. Se
interrumpe de inmediato ante un fin real. Después se obtiene un frame gris de
84×84 y se añade al stack de cuatro. Es una aproximación temporal, no garantía
de un estado completamente Markoviano. No hay frame max-pooling ni LSTM.
El límite externo cuenta **decisiones**, no frames.

El visor conserva la imagen RGB 240×256. Las imágenes se copian porque el
emulador entrega vistas de memoria nativa que mutan al avanzar. El replay guarda
`uint8` en CPU; el modelo convierte a float y divide por 255 solo el batch usado.

## Adaptador NES y recompensa

Este adaptador propio sustituye la propuesta inicial de `gym-super-mario-bros`
para usar la ROM externa. Su recompensa es deliberadamente más pequeña que la
shaping completa de ese paquete. No se presenta como su recompensa nativa.

Por **frame**:

```text
progress = nuevos píxeles más allá del máximo anterior (0 si el salto supera 5)
time = decremento del reloj del juego (0 si aumenta)
death = -25 al morir
completion = +50 al llegar a la bandera
reward = clip(progress + time + death + completion, -15, +15)
```

No se recompensa sobrevivir, retroceder y volver al mismo punto, ni coleccionar
monedas. La muerte y el éxito terminan la única vida; el límite del experimento
es truncación. El timeout interno de Mario termina mediante su muerte normal.

Direcciones RAM de SMB1 utilizadas:

| Campo | Dirección |
|---|---|
| X | `0x006d * 256 + 0x0086` |
| Estado jugador | `0x000e` (normal 8; muerto 6; muriendo 11) |
| Página Y / caída | `0x00b5` (>1 indica caída bajo viewport) |
| Tipo enemigo | `0x0016..0x001a` (bandera 0x31) |
| Deslizamiento bandera | `0x001d` (3) |
| Modo y task de motor | `0x0770`, `0x0772` |
| Mundo y nivel | `0x075f`, `0x075c` |
| Reloj decimal | `0x07f8..0x07fa` |

Las semánticas se contrastaron con el [adaptador original del entorno](https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/smb_env.py)
y se verificaron con la ROM proporcionada. No se copió el adaptador completo.

El arranque pulsa START con límite de 3.600 frames. Espera modo juego, motor
activo task 3, jugador normal, primer tick del reloj y cuatro frames de
estabilización visual. Solo entonces guarda el snapshot inicial. Cada reset
restaura ese snapshot interno y reconstruye todo el stack. No hay RAM hacking,
selección arbitraria de nivel ni save states persistentes entre versiones.

## Aprendizaje

```text
Q_pred = Q_online(observation)[action]
y = reward + gamma * (1 - terminated) * max Q_target(next_observation)
loss = Huber(Q_pred, y)
```

El target no tiene gradientes. La online aprende con Adam y clipping de norma
10; la target se copia cada N **updates**. `truncated` se guarda pero no anula
bootstrap. Se guarda la observación final antes del reset, nunca la del próximo
episodio. Referencias: [DQN original](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf),
[tutorial PyTorch](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
y [límites Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

La evaluación no consume RNG del entrenamiento, no usa replay y ejecuta un
entorno diferente. Esta red no tiene dropout/batchnorm; aun así se cambia a
modo eval y se restaura train explícitamente después.

## Visor y evidencia

Un tick avanza una decisión en cada panel activo. Los paneles terminados quedan
congelados; no se reinician silenciosamente. La selección de políticas reinicia
todos los paneles. Las Q mostradas corresponden al estado en que se eligió la
**última acción**, no a una predicción de éxito.

No hay streaming externo, websocket ni un entrenador en background. El
navegador solicita ticks y muestra PNG codificados. El servidor serializa
mutaciones, sirve únicamente rutas permitidas, rechaza Host/Origin ajenos y
escucha en loopback. No está pensado para publicarse en Internet.

Los MP4 contienen un frame por decisión, no todos los frames de NES ni audio.
Las pausas no agregan frames. El JSON asociado conserva la identidad y resultados.
La reproducción real depende del rendimiento de inferencia/PNG del equipo.

## Límites explícitos

- No se promete completar 1-1 con entrenamiento corto ni generalizar a otros niveles.
- Solo se ha verificado la ROM del usuario registrada por su SHA-256.
- Reset fijo: seeds de evaluación distintas no crean diversidad de estados NES.
- Replay de stacks completos: legibilidad a cambio de RAM (~1,13 GB / 20.000).
- `--resume` permite continuar online/target/Adam y contadores en una salida
  nueva, heredando configuración salvo flags explícitos. `--steps` es objetivo
  total global; epsilon no vuelve a0/1 por reiniciar el proceso.
- Checkpoints no permiten resume exacto: replay, partida y RNG se reinician;
  se recoge warmup de experiencias nuevas antes de más updates. Episodios y
  selección best pertenecen a la continuación; se registra procedencia/SHA.
- CPU predeterminado; GPU requiere otro entorno PyTorch (`.venv-cuda`, ver README)
  y no es reproducible bit a bit por cuDNN.
- Sin sonido, multijugador en un mismo mundo, Double DQN, curriculum ni entrenamiento
  distribuido. `--envs` paraleliza solo la recolección; hay un único aprendiz.

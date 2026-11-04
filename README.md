# Mario · RL Lab

Un **DQN explícito y legible** para Super Mario Bros. de NES (nivel 1-1), con un
visor local para comparar agentes en partidas independientes. No utiliza servicios
externos, un framework RL, ni una recreación del juego para la ejecución real.

## Empezar: el visor

El proyecto y su `.venv` ya están preparados en esta máquina. Desde el directorio
`/home/creep/workshop/rl-mario-bros`:

```bash
.venv/bin/python -m mario_rl compare \
  --rom .local/roms/super-mario-bros.nes
```

Abre **http://127.0.0.1:8765**. Inicialmente hay un agente aleatorio, uno que corre
a la derecha y un DQN **sin entrenar**. La comparación empieza pausada.

- **Reproducir / Pausar:** avanza los agentes activos, sin aprender.
- **Paso:** una decisión por agente; hasta cuatro frames de NES por decisión.
- **Reiniciar:** mismo estado inicial y seed seleccionada en todos los paneles.
- **Selector de agente:** cambia la política y reinicia toda la comparación.
- **Velocidad:** reproducción de 0,25× a 4×; 1× corresponde a 15 decisiones/s
  con el action repeat predeterminado de cuatro frames. El rendimiento puede ser menor.
- **Grabar MP4:** captura un mosaico a partir de los RGB reales, sin audio.
  Detén la grabación antes de cambiar agentes o reiniciar. Límite: 1.800 frames.
- **Reporte:** descarga JSON con seed, entorno, políticas y métricas.

No son tres Marios dentro del mismo mundo: son tres emuladores separados.
La pantalla a color del visor es distinta de la entrada gris del modelo.

### Comparar checkpoints

Los checkpoints se añaden explícitamente; el visor no explora otros directorios:

```bash
.venv/bin/python -m mario_rl compare \
  --rom .local/roms/super-mario-bros.nes \
  --checkpoint runs/nes-validation-final/step-000000512.pt \
  --checkpoint runs/nes-validation-final/final.pt \
  --agents random checkpoint:0 checkpoint:1
```

Estos son checkpoints de una **validación corta**, no agentes capaces de completar
el juego. El selector siempre distingue un DQN sin entrenar de un checkpoint.

## Instalar en otra máquina

Requisitos: **Python 3.13 o 3.14**, [uv](https://docs.astral.sh/uv/) y, para grabar
MP4, **ffmpeg**. El `uv.lock` fija las versiones verificadas y usa PyTorch CPU
para no instalar varios GB de bibliotecas CUDA.

```bash
uv sync --extra nes
```

Para trabajar solo con la infraestructura sintética: `uv sync` (sin `nes-py`).
Para CUDA, usa un entorno separado con una distribución PyTorch apropiada;
el índice configurado en este proyecto instala CPU.

### ROM externa, nunca incluida

Debes proporcionar tu propio archivo `.nes` obtenido legalmente. Este proyecto
**no descarga, publica ni incorpora ROMs**. Solo soporta SMB1 estándar NROM en
1-1; no otros títulos, hacks, niveles arbitrarios ni estados guardados externos.

En esta máquina, el usuario proporcionó el ZIP
`/home/creep/Downloads/Super Mario Bros. (World).zip`. Se extrajo su única ROM a
`.local/roms/super-mario-bros.nes` sin modificar el ZIP. `.local/` y `*.nes` están
ignorados. No distribuyas esa carpeta.

Para otra ROM local:

```bash
.venv/bin/python -m mario_rl compare --rom /ruta/absoluta/a/smb1.nes
```

Una ROM ausente o inválida produce un error: **jamás se sustituye automáticamente
por el fixture**.

## Entrenar

Un experimento real, manual y potencialmente largo:

```bash
.venv/bin/python -m mario_rl train \
  --rom .local/roms/super-mario-bros.nes \
  --steps 1000000 --epsilon-decay 500000 \
  --checkpoint-every 25000 --eval-every 25000 --eval-episodes 1 \
  --seed 7 --output runs/dqn-seed-7
```

### Progreso en la terminal

La barra **tqdm** está activada por defecto. Muestra porcentaje, decisiones
completadas/total, tiempo transcurrido, tiempo restante estimado y velocidad.
El postfix incluye episodio, updates, epsilon, última pérdida, retorno acumulado
del episodio (`reward`) y progreso máximo (`x`). Antes del primer update, la
pérdida aparece como `-`; durante las evaluaciones cambia a **Evaluando**.
El tiempo restante es una estimación, no una promesa de convergencia.
En terminales estrechas, tqdm recorta el postfix: amplía la terminal para leer
todas las métricas. Los CSV conservan los valores aunque no quepan en pantalla.

La barra cuenta **decisiones, no frames ni épocas**. El visor no entrena.
Todo progreso y mensajes van a **stderr**, dejando el JSON final en stdout
limpio para scripts. Para desactivar solo la barra, añade `--no-progress`;
los mensajes de evaluación/guardado y los CSV se conservan.

### Checkpoints automáticos

Cada guardado anuncia su ruta absoluta y decisión en la terminal. Se conserva
un snapshot inicial, uno cada `--checkpoint-every` decisiones (25.000 por defecto),
el mejor en evaluación, el final y, si detienes con Ctrl+C, el interrumpido.
`best.pt` prioriza éxito y después progreso; `final.pt` no necesariamente es mejor.
Las escrituras son atómicas, de modo que el visor no carga archivos a medio guardar.
Puedes cargar un checkpoint periódico en otro proceso mientras continúa el
entrenamiento; el visor conserva el modelo cargado, no lo actualiza automáticamente.

El entrenamiento no abre ventanas. Conserva:

| Archivo | Contenido |
|---|---|
| `config.json` | Hiperparámetros, identidad del entorno, ROM SHA-256 y versiones. |
| `episodes.csv` | Retorno, progreso, éxito, muerte, truncación, decisiones y frames. |
| `updates.csv` | Pérdida, número de actualización y epsilon. |
| `evaluation.csv` | Evaluación greedy en un entorno separado. |
| `step-*.pt` | Snapshot inicial y checkpoints periódicos. |
| `best.pt` | Mejor éxito; desempate por progreso. No significa competencia. |
| `final.pt` | Último modelo. |
| `summary.json` | Resultado del entrenamiento y advertencias. |

El directorio de salida debe ser nuevo o estar vacío: no se sobrescriben
experimentos. Ctrl+C guarda `interrupted.pt`. Los checkpoints permiten inferencia,
comparación y **continuar el aprendizaje** con `--resume`, con los límites siguientes.

### Continuar desde un checkpoint

Para recuperar el experimento existente de esta máquina sin modificarlo:

```bash
P=/home/creep/workshop/rl-mario-bros
"$P/.venv/bin/python" -m mario_rl train \
  --rom "$P/.local/roms/super-mario-bros.nes" \
  --resume "$P/runs/dqn-progress-7/step-000025000.pt" \
  --steps 1000000 \
  --output "$P/runs/dqn-progress-7-resumed"
```

- **`--steps` es el objetivo TOTAL:** desde 25.000 hasta 1.000.000 se ejecutan
  975.000 decisiones nuevas, no otro millón. Debe ser mayor que el contador guardado.
  Si se omite, se conserva el objetivo del checkpoint.
- Se restauran **online, target, Adam y contadores de decisiones/updates**.
  La barra comienza en el progreso guardado y epsilon continúa su calendario global.
- Los hiperparámetros (`epsilon-decay`, gamma, seed, batch, etc.) y límites del
  entorno se heredan salvo flags explícitos. No hace falta repetir `--epsilon-decay
  500000`. `--rom` o `--demo` siguen siendo explícitos. Para mover un checkpoint
  entre GPU y CPU, añade `--device cpu` o `--device cuda`. `--envs` también se
  hereda y puede cambiarse al reanudar: un checkpoint de 1 emulador continúa con 8.
- Replay, partida e historial RNG **se reinician**. Antes de nuevos updates se
  recogen `warmup` decisiones nuevas y un batch completo, manteniendo epsilon
  global (no se fuerza a1). Si el presupuesto nuevo es menor que ese refill,
  se conserva el modelo, pero aún no habrá nuevos updates.
- `--output` debe ser **nuevo/vacío**, incluso con resume. El origen no se toca;
  no se continúa automáticamente por el solo hecho de elegir una carpeta existente.
  Los episodios, CSV y selección `best.pt` pertenecen a la nueva ejecución.
- `config.json` y `summary.json` registran ruta/SHA del origen, paso inicial y
  reinicios de estado. El snapshot inicial se llama `step-000025000.pt`, no paso0;
  `summary.steps_executed` distingue trabajo nuevo del contador global.

Esto **no reproduce exactamente la ejecución interrumpida**: faltan replay,
estado del emulador y RNG previos. El nuevo `best.pt` es el mejor de las evaluaciones
de la continuación; conserva el `best.pt` original para comparar. Para retomar
una continuación interrumpida, usa su `interrupted.pt` y otra salida nueva.

Valores predeterminados: gamma 0,99; Adam 1e-4; batch 32; replay 20.000;
warmup 5.000 decisiones; epsilon 1 → 0,05 durante 200.000 decisiones **después**
del warmup; un update cada cuatro decisiones; copia target cada 1.000 updates;
evaluación y checkpoint cada 25.000 decisiones. Consulta todos los flags:

```bash
.venv/bin/python -m mario_rl train --help
```

**Recursos:** el replay guarda ambos stacks `uint8` en arrays preasignados:
~1,13 GB de píxeles para 20.000 transiciones (~56 KB cada una). La CNN puede
entrenarse en CPU, pero completar el nivel no tiene un plazo garantizado ni está
demostrado con una prueba corta.

### Acelerar: emuladores en paralelo y GPU

Cada decisión cuesta casi lo mismo en el emulador (~4,5 ms) que en la red
(~3,6 ms por decisión en CPU). Por eso hacen falta dos cambios a la vez:

- **`--envs N`** lanza N emuladores, cada uno en su propio proceso y núcleo.
  Cada ronda la red elige N acciones en un solo forward y los N emuladores
  avanzan a la vez. El aprendizaje, los checkpoints y los CSV siguen contando
  **decisiones**: con `--envs 8` se sigue haciendo un update cada 4 decisiones.
  `--envs 1` (predeterminado) es exactamente el bucle original, verificado con
  pesos finales idénticos. `episodes.csv` añade la columna `env`.
- **`--device cuda`** entrena la red en la GPU. Requiere un PyTorch con CUDA,
  instalado en un entorno separado para no alterar `.venv`:

```bash
uv venv --python 3.13 .venv-cuda
uv pip install --python .venv-cuda/bin/python "torch==2.14.1" \
  --index-url https://download.pytorch.org/whl/cu132
uv pip install --python .venv-cuda/bin/python -e ".[nes]" --no-deps
uv pip install --python .venv-cuda/bin/python "numpy>=2.1,<3" "pillow>=11,<13" \
  "gymnasium>=1.1,<2" "tqdm>=4.67,<5" "nes-py==9.0.1"
```

Usa ese intérprete directamente; `uv run` sincroniza `.venv`, no `.venv-cuda`:

```bash
.venv-cuda/bin/python -m mario_rl train \
  --rom .local/roms/super-mario-bros.nes \
  --envs 8 --device cuda \
  --steps 1000000 --epsilon-decay 500000 \
  --checkpoint-every 25000 --eval-every 25000 --eval-episodes 1 \
  --seed 7 --output runs/dqn-seed-7-gpu
```

Medido en esta máquina (12 núcleos, RTX 4070 SUPER), 5.000 decisiones con
aprendizaje activo:

| Configuración | Decisiones/s | RAM del árbol de procesos* |
|---|---|---|
| CPU, `--envs 1` | 104 | 0,8 GB |
| CPU, `--envs 8` | 180 | 1,2 GB |
| GPU, `--envs 1` | 186 | 2,1 GB |
| GPU, `--envs 8` | 600 | 2,5 GB |
| GPU, `--envs 10` | 679 | 2,6 GB |

\* Con el replay casi vacío; súmale ~1,1 GB cuando se llena (20.000).

En solo CPU, más emuladores apenas ayudan: el cuello de botella pasa a ser la
red. Deja dos núcleos libres (`--envs` ≤ núcleos − 2); el CLI avisa si no.
**Vigila la RAM:** si el sistema supera su umbral de memoria, systemd-oomd puede
matar la sesión de terminal completa sin guardar `interrupted.pt`.

En GPU el aprendizaje no es reproducible bit a bit: cuDNN acumula gradientes
de convolución con operaciones atómicas. En CPU sí lo es, también con `--envs N`.

**Evaluación en NES:** el reset parte siempre del mismo estado y la política
evaluada es determinista, así que todos los episodios de una evaluación son
idénticos. `--eval-episodes` mayor que 1 solo repite la misma partida; la
variación de `x` entre evaluaciones viene de los cambios de pesos.

## Evaluar

```bash
.venv/bin/python -m mario_rl evaluate \
  --rom .local/roms/super-mario-bros.nes \
  --checkpoint runs/dqn-seed-7/final.pt \
  --episodes 10 --output runs/evaluation-seed-7.json
```

Sin checkpoint puedes elegir `--policy random`, `right` o `untrained`. No se
realizan actualizaciones ni se llena replay. Se rechazan checkpoints con otra
ROM, modo demo/NES, preprocesamiento, reset, recompensa o acciones.

**Interpretación académica:** el reset NES es determinista. Cambiar la seed de
reset no crea nuevos niveles ni situaciones iniciales; repetir una política
greedy desde el mismo inicio produce el mismo recorrido. Para estudiar
variabilidad, entrena modelos con varias seeds y presupuestos comparables.
Este MVP no demuestra generalización a otros niveles.

## Verificación reproducible

El contrato E2E y los modos de fallo se escribieron antes de implementar. No se
utilizan tests unitarios que reafirmen detalles de implementación.

```bash
.venv/bin/python tests/e2e.py \
  --rom /home/creep/workshop/rl-mario-bros/.local/roms/super-mario-bros.nes
```

El harness ejecuta CLI y HTTP completos: entrenamiento, evaluación en procesos
nuevos, carga de checkpoints, interacción NES, controles del comparador,
congelación al terminar, guardas de origen, PNG, JSON y MP4. Cada ejecución
conserva evidencia independiente en `artifacts/e2e/run-*/`; el último reporte
queda en `artifacts/e2e/report.json`. Requiere ffmpeg y ffprobe.

Para verificar específicamente la barra y checkpoints (sin servidor HTTP):

```bash
.venv/bin/python tests/training_progress_e2e.py \
  --rom /home/creep/workshop/rl-mario-bros/.local/roms/super-mario-bros.nes
```

Comprueba stdout JSON limpio, progreso al 100%, carga de todos los tipos de
checkpoint en procesos nuevos, Ctrl+C y aprendizaje idéntico con/sin barra.
Conserva logs, checkpoints, CSV y reporte en `artifacts/training-progress/`.

Para verificar resume (restauración exacta de pesos/Adam, configuración heredada,
presupuesto global, protección del origen y Ctrl+C):

```bash
.venv/bin/python tests/resume_e2e.py \
  --rom /home/creep/workshop/rl-mario-bros/.local/roms/super-mario-bros.nes \
  --checkpoint /home/creep/workshop/rl-mario-bros/runs/dqn-progress-7/step-000025000.pt
```

El checkpoint del usuario es opcional y se lee sin modificarlo. La continuación
de prueba solo ejecuta 40 decisiones nuevas en `artifacts/resume/run-*/`; no lanza
el entrenamiento largo. Reporte y comandos quedan en `artifacts/resume/report.json`.

Para verificar el modo paralelo (pasos exactos, CSV, reanudar con `--envs`,
Ctrl+C sobre todo el grupo de procesos y ausencia de procesos huérfanos):

```bash
.venv/bin/python tests/parallel_e2e.py --rom .local/roms/super-mario-bros.nes
.venv-cuda/bin/python tests/parallel_e2e.py --rom .local/roms/super-mario-bros.nes --device cuda
```

Para una comprobación breve del emulador:

```bash
.venv/bin/python -m mario_rl smoke \
  --rom .local/roms/super-mario-bros.nes --output artifacts/my-nes-smoke
```

Sin ROM, usa **explícitamente** `--demo`. Es un entorno geométrico creado para
verificar infraestructura, rotulado **NOT MARIO / NOT NES** en pantalla, vídeo
y reportes. Sus resultados no cuentan como aprendizaje de Mario.

## Leer el código

Empieza por `training.py` y sigue el flujo:

```text
Environment → Preprocessing → Agent → Replay → Learn → Evaluate → Checkpoint
                                      ↓
                Policies → ComparisonRunner → HTTP server → Browser viewer
```

La CNN, el Bellman target, el replay y el ciclo de entrenamiento están escritos
explícitamente. El visor usa HTML/CSS/JavaScript sin dependencias frontend y el
servidor HTTP de la biblioteca estándar, ligado exclusivamente a 127.0.0.1.

Consulta [arquitectura](docs/ARCHITECTURE.md), [verificación](docs/VERIFICATION.md)
y [registro de trabajo](docs/WORK.md) para decisiones, evidencia y limitaciones.

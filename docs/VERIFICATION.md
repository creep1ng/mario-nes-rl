# Verificación de la implementación

## Reanudación verificada (13 de octubre de 2026)

**Passed:** resume E2E, progreso E2E y suite completa sintética/NES/HTTP/MP4.
No se alteró la ROM ni el checkpoint del usuario, ni se lanzó entrenamiento largo.

| Comprobación | Evidencia |
|---|---|
| Online/target/Adam/contadores restaurados | Snapshot inicial de la continuación idéntico al origen, incluido estado Adam. |
| Presupuesto total | 40 -> --steps56 ejecuta16 decisiones nuevas y3 updates, en fixture y NES. |
| Configuración heredada | Gamma0.97, seed11, decay100, repeat2, límite16 y otros valores no explícitos se conservan. |
| Epsilon/replay | Primer nuevo update48 después de8 experiencias nuevas; epsilon global correcto, sin reset artificial. |
| Validaciones | Objetivo <= checkpoint, archivo ausente/corrupto, action_repeat incompatible y output de origen rechazados antes de escribir. |
| Ctrl+C en continuación | Exit130, interrupted.pt desde paso64 y evaluación desde proceso nuevo. |
| Origen intacto | Hashes de archivos del experimento base iguales antes/después. |
| Checkpoint del usuario | 25.000 decisiones/5.001 updates restaurados;40 decisiones de prueba, SHA original intacto. |
| No regresión | E2E de progreso y completo con NES/HTTP/MP4 pasaron otra vez. |
| Sintaxis | compileall de src/tests completado; no nuevas dependencias. |

Reproducir desde el proyecto:

```bash
/home/creep/workshop/rl-mario-bros/.venv/bin/python \
  /home/creep/workshop/rl-mario-bros/tests/resume_e2e.py \
  --rom /home/creep/workshop/rl-mario-bros/.local/roms/super-mario-bros.nes \
  --checkpoint /home/creep/workshop/rl-mario-bros/runs/dqn-progress-7/step-000025000.pt
```

La continuación del checkpoint del usuario tuvo presupuesto40 < refill5000:
restauró5001 updates pero no hizo nuevos updates en esa prueba. Las continuaciones
fixture/NES con refill8 sí verifican entrenamiento posterior y crecimiento de
pasos internos de Adam. No se probó GPU ni reconstrucción exacta de replay/NES/RNG;
se reinician explícitamente. Tampoco se hizo nueva QA de navegador para este cambio CLI.

Evidencia actual:
- Resume: `artifacts/resume/report.json`, directorio `artifacts/resume/run-1791912630436282379`.
- Progreso: `artifacts/training-progress/report.json`, directorio `artifacts/training-progress/run-1791912712256679231`.
- Suite completa: `artifacts/e2e/report.json` (directorio indicado en su campo evidence_directory).
- SHA del checkpoint usuario leído/verificado: `24b2e267975473d1d85a0ab924f71255c02fe0cd7257625e5fb8ca89d0231a93`.
- Fuente actual verificada en los tres reportes: `c3c1e0c248b1abaab52ceb0199ef6b3f9b7d2b6bfb7360e919c5a71acc8460be`.
- Logs stdout/stderr, comandos exactos, CSV, evaluaciones y snapshots conservados
  por ejecución. Las pruebas de error no crean directorios de continuación.

El contrato E2E se escribió antes de implementar y falló inicialmente porque
--resume no existía. La corrección no cambia el formato v1 de checkpoints; añade
carga de Adam y contadores, herencia de configuración y presupuesto total.
Los contadores guardados ahora representan ciclos de decisión completados.

## Ampliación anterior: tqdm y checkpoints (9 de octubre de 2026)

Las pruebas se repitieron después de añadir progreso de entrenamiento. La barra
cuenta decisiones, muestra métricas y distingue evaluación; mensajes de guardado
y evaluación se escriben en stderr, sin contaminar el JSON final de stdout.

| Comprobación | Resultado |
|---|---|
| CLI con/sin barra, fixture y NES | 48 decisiones, 11 updates por ejecución; CSV de episodios, updates y evaluación idénticos. |
| Carga de snapshots en procesos nuevos | Inicial, 16, 32, 48, best, final e interrupted cargan en ambos modos. |
| Ctrl+C/SIGINT | exit130; barra no señala 100%, interrupted.pt cargable; no quedan .tmp. |
| stdout/stderr | JSON final parseable; barra, evaluación y avisos en stderr. |
| Compatibilidad anterior | Checkpoint NES512 de runs/nes-validation-final cargado y evaluado. |
| PTY real NES | 64 decisiones, 15 updates; transcript con porcentaje, ETA y fases. Postfix recortado a 80 columnas, documentado. |
| PTY real ancho | 32 decisiones, 7 updates; a 160 columnas todos los campos del postfix visibles. |
| Suite completa | passed: synthetic and real NES E2E; incluye HTTP y MP4. |
| Dependencias y sintaxis | uv sync --extra nes --locked --offline y compileall completados. tqdm4.70.1 declarado explícitamente. |

Comando específico reproducible (no requiere servidor HTTP):

```bash
/home/creep/workshop/rl-mario-bros/.venv/bin/python \
  /home/creep/workshop/rl-mario-bros/tests/training_progress_e2e.py \
  --rom /home/creep/workshop/rl-mario-bros/.local/roms/super-mario-bros.nes
```

Evidencia de esta ampliación:
- Reporte de progreso: `/home/creep/workshop/rl-mario-bros/artifacts/training-progress/report.json`.
- Logs, checkpoints, CSV y evaluaciones: `artifacts/training-progress/run-1791583556424201410`.
- Transcript PTY: `artifacts/training-progress/terminal/nes-terminal.log`.
- Transcript ancho: `artifacts/training-progress/terminal/nes-wide-tty-terminal.log`.
- E2E completo actualizado: `artifacts/e2e/run-1791583604998506333`; último reporte en `artifacts/e2e/report.json`.
- Source SHA-256 de esa ampliación: `796b7acb12734c46c17973802c5cddc83124a9528ab67975e5d93d675a2a4a35`.
- Cada config.json conserva versiones/hilos del proceso CLI; runtime del reporte
  específico describe el harness, no debe confundirse con sus procesos hijos.

El contrato específico se escribió antes del cambio. Su ejecución inicial falló
al parsear stdout: los prints de evaluación mezclaban texto con JSON. Se corrigió
usando tqdm.write en stderr. No hubo cambio de algoritmo, recompensa ni formato
de checkpoint. No se añade resume ni se demuestra convergencia. No se repitió
QA de navegador para este cambio de terminal; la evidencia visual siguiente
pertenece a la primera entrega.

## Evidencia de la primera entrega

## Resultado

**passed: synthetic and real NES E2E**. QA de navegador desktop también completada.
La ROM real fue proporcionada por el usuario; no se descargó ninguna ROM.
Esto verifica el sistema, **no** que el modelo sea competente en Mario.

## Reproducir

Desde `/home/creep/workshop/rl-mario-bros`:

```bash
.venv/bin/python tests/e2e.py --rom /home/creep/workshop/rl-mario-bros/.local/roms/super-mario-bros.nes
```

Requisitos: `.venv` instalada con `uv sync --extra nes`, ROM SMB1 válida,
ffmpeg/ffprobe, permiso para servidores temporales en 127.0.0.1. El harness crea
un directorio nuevo en cada ejecución y no sobrescribe experimentos de entrenamiento.

## Evidencia

- Último reporte: `artifacts/e2e/report.json`.
- Evidencia completa: `/home/creep/workshop/rl-mario-bros/artifacts/e2e/run-1791581482987162981`.
- Source SHA-256 (Python + HTML/CSS/JS): `23fe97d308d93300769518a5939ac324d1e6d0665c34bcd7e41b3eb71f44a404`.
- ROM SHA-256: `f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de`.
- Logs de comandos/exit codes, CSV, evaluaciones, checkpoints, PNG y MP4 están
  dentro del directorio de evidencia.
- Vídeo E2E NES: `/home/creep/workshop/rl-mario-bros/artifacts/e2e/run-1791581482987162981/nes-viewer/recordings/comparison-1791581628462923946.mp4`.
- Vídeo E2E SHA-256: `12534c510368e1ce8bf8e108d6c28d177fc03d5948aa64e566df346633acebf8`.
- Vídeo comprobado con ffprobe: H.264, 1536×584, 129 frames, sin audio.
- QA de navegador: `artifacts/visual-qa/report.json` y `viewer-desktop.jpg`.
- Demostración de los checkpoints 512/1024: `artifacts/visual-qa/comparison-real.mp4`
  y JSON asociado; 51 frames (inicio + 50 decisiones), sin audio.

## Comprobaciones realizadas

| Recorrido | Resultado |
|---|---|
| ROM ausente / checkpoint corrupto | Error claro, sin fallback a demo. |
| Fixture explícito | Éxito, muerte, truncación y corte de action repeat observados. |
| Entrenamiento fixture | 48 decisiones, 11 updates, pérdidas finitas. |
| NES real | Inicio 1-1, movimiento, RGB del emulador, muerte, snapshots independientes. |
| Entrenamiento E2E NES | 256 decisiones, 63 updates, pérdidas finitas. |
| Carga en procesos nuevos | Evaluaciones reproducibles con mismo checkpoint/seed. |
| Checkpoint de NES en demo | Rechazado por incompatibilidad de entorno. |
| Comparador HTTP | GET no avanza; step, selección, reset y congelación terminal correctos. |
| Modelos congelados | Hashes de checkpoints no cambian tras evaluación/visor. |
| JSON / MP4 | Exportación con modo/seed/resultados; contenedor H.264 verificable. |
| Host / Origin | Origen ajeno rechazado (403); catálogo no permite rutas arbitrarias. |
| UI desktop real | Paso, play/pause, velocidad, selección, seed, Q, descarga y grabación. |
| Console JS | Sin errores/warnings durante la comprobación final. |
| Sintaxis | `python -m compileall -q src tests` y `node --check .../app.js` pasaron. |

## Validación adicional, no benchmark

Se ejecutó un entrenamiento corto real de 1024 decisiones y
241 actualizaciones. Se guardó en `runs/nes-validation-final`.
Pérdida final finita: 3.4006855487823486. Éxito observado: **0%**.
El checkpoint 512 avanzó hasta X=297 en evaluación; el último aprendió una
política que se queda a la izquierda, X máxima=40. No se oculta la degradación
ni se etiqueta como un agente que sabe completar Mario.

## Fallos encontrados y correcciones

1. El primer socket bajo sandbox produjo EPERM. La prueba local se ejecutó con
   permiso para loopback. Un revisor automático agotó su plazo una vez; el
   reintento permitido funcionó. No se cambiaron ajustes de seguridad.
2. El primer snapshot NES guardaba el menú: el modo cambiaba antes del motor.
   Se verificó RAM + imágenes, se esperó task3/reloj/PPU y se versionó reset.
   Los experimentos viejos están marcados `OBSOLETE.md` y sus checkpoints se rechazan.
3. Una respuesta asíncrona reescribía la seed de la próxima sesión. Se conservó
   el input mientras la seed remota no cambia y se bloqueó durante operaciones;
   QA repitió selección → seed19 → reset y confirmó seed19 en pantalla.
4. Un locator de automatización de velocidad no coincidió con el label exacto;
   se usó el ID observado del control. No fue un fallo de la aplicación.

## Límites / checks no realizados

- No se completó 1-1 con el agente real, ni se comprobó llegada real a la bandera.
  El fixture cubre la ruta de éxito; la detección RAM real todavía necesita un
  intento exitoso como evidencia adicional.
- No se hizo entrenamiento de un millón de decisiones, estudio de varias seeds,
  convergencia, generalización o benchmark de rendimiento/GPU.
- No se validó un viewport móvil; la inspección fue desktop.
- No hay tests unitarios, resume exacto, audio ni multiplayer en un mismo mundo.
- No se crearon commits: el directorio inicial no era un repositorio Git utilizable.

## Estado de entrega

Visor local en `http://127.0.0.1:8765`, pausado en decisión 50, con aleatorio,
checkpoint512 y checkpoint1024. El servidor no entrena ni avanza solo.
Ctrl+C detiene el servidor. Para volver a abrirlo, usa el comando de README.

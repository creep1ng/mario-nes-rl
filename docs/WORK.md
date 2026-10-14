# Seguimiento de implementación

## Alcance autorizado
DQN académico legible para Super Mario Bros. NES, nivel 1-1, más visor
comparativo de agentes en instancias independientes. No descargar ni distribuir ROMs.
El entorno sintético sirve exclusivamente para verificar la infraestructura y nunca
se presenta como Mario. No se promete un agente competente sin entrenamiento real.

## Recuperación y entorno
- El archivo solicitado `/home/creep/.codex/instructions/odd-workflow.md` no existe.
  Se usa este registro local de tareas, decisiones y evidencia como alternativa.
- Directorio inicial vacío; no es un repositorio Git utilizable. No crear commits.
- Python del sistema: 3.14.7; dependencias en `.venv`, caché uv en `/tmp`.
- ROM proporcionada por el usuario: `/home/creep/Downloads/Super Mario Bros. (World).zip`.
  Extraída sin modificar el ZIP a `.local/roms/super-mario-bros.nes` (ignorada).

## Unidades de trabajo
- [x] 1. Contrato E2E y modos de fallo, antes de implementar.
- [x] 2. Entorno real con ROM externa y fixture sintético explícito.
- [x] 3. CNN, replay, DQN, entrenamiento, evaluación y checkpoints.
- [x] 4. Comparador, interfaz local, controles y exportación.
- [x] 5. Ejecutar E2E, inspeccionar interfaz, conservar reporte y vídeo.
- [x] 6. Documentación de uso, límites y reproducción.

## Modos de fallo y aceptación (definidos antes del código)
1. ROM ausente o inválida: error accionable; jamás sustitución silenciosa por demo.
2. Frames nativos mutables: copiar antes de almacenar o entregar al visor.
3. Historia visual de otro episodio: reset reconstruye cuatro frames iniciales.
4. Fin durante action repeat: no ejecutar frames después de muerte/éxito/límite.
5. Truncación confundida con terminal: bootstrap solo se corta con terminated.
6. Entrenamiento desconectado: ejecución CLI corta debe producir pérdidas finitas,
   actualizaciones, checkpoints y evaluación cargable en un proceso nuevo.
7. Checkpoint incompatible/corrupto: rechazar con explicación; weights_only=True.
8. Comparación injusta: mismo nivel, seed inicial, actions y límite por agente;
   un agente terminado se congela hasta reinicio explícito.
9. Estado cambia al leerlo: GET no avanza el juego; un step = una decisión por panel activo.
10. Evaluación modifica modelos: no optimizador ni replay en comparación/evaluación.
11. Error HTTP o carreras: serializar operaciones; controles no solapan requests.
12. Exportación falsa: vídeo y JSON incluyen modo, políticas, seeds y métricas;
    fallo de ffmpeg visible; ninguna grabación se etiqueta como Mario si es demo.
13. UI rota: comprobar pausa, paso, velocidad, reinicio, selector y descarga,
    además de inspeccionar screenshot en navegador.
14. La demo pasa pero Mario no se ejecutó: reporte debe marcar la validación NES
    como pendiente, no declarar integración Mario verificada.

## Evidencia y rollback
Cada unidad se entrega sin commits. El paquete nuevo y sus docs pueden retirarse
sin afectar código preexistente (no había ninguno). Comandos, versiones y resultados
se guardarán en `artifacts/e2e/report.json` y en `docs/VERIFICATION.md`.

## Estado actual
- Dependencias instaladas y lockfile creado.
- ROM real ejecutada, movimiento y muerte observados.
- Snapshot inicial corregido: engine task 3 + reloj + cuatro frames de estabilización.
- Entrenamiento NES de validación: 1.024 decisiones, 241 updates, pérdidas finitas,
  éxito 0%; no se presenta como agente competente.
- E2E previo pasó fixture, HTTP y smoke NES. El E2E final añade entrenamiento
  y vídeo NES real. El primer intento de socket bajo sandbox falló (EPERM);
  se ejecuta con permiso para loopback. Un revisor automático agotó su plazo;
  el reintento permitido inició correctamente.
- Visor real abierto en navegador; QA de controles en curso.

## Entrega verificada
- E2E final pasó con fixture y NES real; ver docs/VERIFICATION.md.
- Source hash final coincide con el reporte E2E.
- QA desktop y vídeo real guardados en artifacts/visual-qa.
- Visor 8765 reiniciado con código final y dejado pausado, sin entrenamiento activo.
- Falta entrenar un agente competente y verificar llegada real a bandera; no forma
  parte de la promesa de la validación corta.

## Ampliación: progreso de entrenamiento y checkpoints

Petición autorizada: incluir `tqdm` ("tqdn" en la solicitud) y checkpoints
visibles. Se conserva el formato de checkpoints existente, sin añadir resume
ni cambiar recompensa, algoritmo o visor.

- [x] Contrato CLI/E2E escrito antes de modificar la implementación.
- [x] Barra por decisiones, métricas, fase de evaluación y mensajes de checkpoint.
- [x] Dependencia explícita, `--no-progress` y guía de uso.
- [x] Verificación CLI con fixture y ROM real; evidencia reproducible.

Modos de fallo y aceptación:
1. Barra llega al 100% antes de terminar o cuenta frames como decisiones: la
   salida debe mostrar el presupuesto de decisiones completado, sin cambiar el aprendizaje.
2. Logs/barra contaminan JSON de stdout: todo progreso y mensajes van a stderr.
3. Logs intercalados rompen la barra: usar `tqdm.write`; evaluación identifica su fase.
4. `--no-progress` sigue emitiendo retornos de carro/barra: debe desactivarla,
   manteniendo mensajes y el mismo comportamiento del agente.
5. Checkpoints periódicos/best/final no cargan o quedan a medias: evaluarlos
   desde un proceso nuevo; no debe quedar ningún `.tmp` después de finalizar.
6. Ctrl+C deja la barra abierta o pierde el modelo: finalizar la barra y guardar
   `interrupted.pt` cargable, sin señalar falsamente 100%.
7. Nuevo logging modifica política/updates: comparar ejecuciones seed idéntica
   con barra activada/desactivada.

Una unidad de trabajo, sin commits. Rollback: quitar la barra/mensajes y flag de
training.py/cli.py, dependencia de pyproject.toml/uv.lock y la documentación de
esta ampliación; el formato y capacidad anterior de checkpoints se conservan.

Resultado de la ampliación:
- E2E específico pasó con fixture y NES: 48 decisiones/11 updates por ejecución,
  siete snapshots cargados por modo, CSV idénticos con/sin barra, SIGINT exit130
  y checkpoint interrumpido cargable. Reporte: artifacts/training-progress/report.json.
- E2E completo pasó otra vez con NES/HTTP/MP4; artifacts/e2e/report.json.
- Fuente verificada: 796b7acb12734c46c17973802c5cddc83124a9528ab67975e5d93d675a2a4a35.
- Transcript PTY real guardado en artifacts/training-progress/terminal;
  a 80 columnas tqdm recorta el postfix, documentado en README.
- Dependencias sincronizadas con --locked --offline y sintaxis Python comprobada.
- Checkpoint NES512 anterior cargó sin cambios de formato. No se lanzó un
  entrenamiento largo ni se modificó recompensa, emulador o visor.

## Ampliación autorizada: reanudar desde checkpoint

- [x] Contrato E2E y modos de fallo antes de implementar.
- [x] --resume explícito, restauración Adam/redes/contadores y configuración heredada.
- [x] Salida nueva, origen intacto, presupuesto total y replay nuevo documentados.
- [x] E2E fixture/NES, checkpoint del usuario y pruebas existentes; reporte reproducible.

Decisiones de alcance: `--steps` es el objetivo total global (no pasos adicionales).
Sin flags explícitos se heredan los hiperparámetros y límites del checkpoint.
Replay, partida y RNG se reinician; recopilar warmup de experiencias nuevas antes
de aprender, sin reiniciar epsilon. Episodios y mejor resultado son de la nueva
ejecución. No reanudar automáticamente ni escribir en el directorio de origen.

Modos de fallo / aceptación:
1. Continuación silenciosa desde pesos aleatorios o Adam nuevo: snapshot inicial
   de la continuación debe conservar exactamente online/target/Adam y contador.
2. Defaults CLI pisan epsilon decay/seed/gamma previos: heredar los valores cuando
   no se especifican, registrar overrides explícitos en la nueva configuración.
3. Confundir presupuesto adicional/total: 40 -> --steps56 ejecuta solo16 decisiones;
   objetivo <= checkpoint debe rechazarse antes de escribir.
4. Replay vacío: no aprender hasta recoger warmup/batch de nuevas experiencias.
   Epsilon usa el contador global, no vuelve artificialmente a1.
5. Origen o logs sobrescritos: output nuevo/vacío y hashes del origen intactos.
6. Checkpoint corrupto/incompatible/ausente: error claro sin crear artefactos falsos.
7. Progreso o Ctrl+C vuelven a0: barra comienza en contador restaurado; el nuevo
   interrupted.pt conserva el avance y carga en un proceso independiente.
8. Se promete exactitud inexistente: avisar que replay/NES/RNG no se recuperan y
   guardar procedencia/SHA del checkpoint en config y summary.

Unidad de trabajo sin commits. Rollback: retirar --resume y resolución de defaults
en cli.py, carga training/Adam en agent.py, rama de continuación en training.py y
docs/test de esta ampliación; no tocar checkpoints del usuario ni el guardado previo.

Resultado de resume:
- Contrato anterior falló por --resume desconocido; después pasó en fixture/NES.
- 40 -> 56 ejecutó16 decisiones y3 updates; pesos/target/Adam/contadores del
  snapshot inicial son idénticos al origen. Configuración/límites heredados,
  epsilon global, refill, errores, protección y Ctrl+C64 comprobados.
- El checkpoint real del usuario25k/5001 se recuperó y se ejecutaron40 decisiones
  de prueba en artifacts/resume, con SHA origen intacto. No se lanzó el millón.
- Pasaron también E2E completo NES/HTTP/MP4 y E2E de barra/checkpoints.
- Los tres reportes corresponden a fuente
  c3c1e0c248b1abaab52ceb0199ef6b3f9b7d2b6bfb7360e919c5a71acc8460be.
- Dependencias no cambiaron; compileall pasó. Evidencia en docs/VERIFICATION.md.

'use strict';
const $ = (selector, parent = document) => parent.querySelector(selector);
const actions = ['Nada', 'Derecha', 'Derecha + salto', 'Derecha + correr',
  'Derecha + correr + salto', 'Salto', 'Izquierda'];
const number = new Intl.NumberFormat('es-CO', {maximumFractionDigits: 2});
let state = null;
let playing = false;
let busy = false;
let timer = null;
let histories = [];
let cards = [];

async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function controls() {
  $('#play').textContent = playing ? 'Ⅱ Pausar' : '▶ Reproducir';
  $('#play').classList.toggle('active', playing);
  $('#step').disabled = busy || playing || state?.all_done;
  $('#reset').disabled = busy || state?.recording;
  $('#record').disabled = busy;
  $('#seed').disabled = busy || playing || state?.recording;
  $('#speed').disabled = state?.recording || false;
  cards.forEach(card => $('.policy-select', card).disabled = busy || state?.recording);
  $('#playback-status').textContent = state?.all_done ? 'Intentos finalizados · reinicia para continuar' :
    playing ? 'Reproduciendo · una decisión por tick' : 'Pausado · puedes avanzar paso a paso';
}

function pause() {
  playing = false;
  clearTimeout(timer);
  controls();
}

async function perform(operation) {
  if (busy) return;
  busy = true;
  $('#error').hidden = true;
  controls();
  try { return await operation(); }
  catch (error) {
    pause();
    $('#error').textContent = error.message;
    $('#error').hidden = false;
  } finally { busy = false; controls(); }
}

function buildCards(data) {
  $('#panels').replaceChildren();
  cards = data.panels.map((panel, index) => {
    const card = $('#panel-template').content.firstElementChild.cloneNode(true);
    $('.panel-number', card).textContent = `AGENTE ${String(index + 1).padStart(2, '0')}`;
    const select = $('.policy-select', card);
    for (const policy of data.catalog) {
      const option = document.createElement('option');
      option.value = policy.id;
      option.textContent = policy.label;
      select.append(option);
    }
    select.addEventListener('change', async () => {
      pause();
      const result = await perform(() => api('/api/select', {panel: index, policy: select.value}));
      if (result) render(result);
      else select.value = state.panels[index].policy;
    });
    $('#panels').append(card);
    return card;
  });
}

function render(data) {
  if (!cards.length || cards.length !== data.panels.length) buildCards(data);
  if (!state || data.tick === 0) histories = data.panels.map(() => []);
  if (!state || data.tick !== state.tick || data.tick === 0) {
    data.panels.forEach((panel, index) => {
      histories[index].push(panel.progress_max);
      histories[index] = histories[index].slice(-120);
    });
  }
  const seedChanged = !state || state.seed !== data.seed;
  state = data;
  $('#tick').textContent = number.format(data.tick);
  $('#session-seed').textContent = data.seed;
  // This field is the user's NEXT reset seed. Same-seed responses must not
  // overwrite a value typed while an earlier asynchronous reset was finishing.
  if (seedChanged) $('#seed').value = data.seed;
  $('#agent-count').textContent = data.panels.length;
  $('#environment-title').textContent = data.mode === 'demo' ? 'Fixture sintético · infraestructura' : 'Super Mario Bros. · NES real';
  $('#demo-warning').hidden = data.mode !== 'demo';
  const maxProgress = Math.max(1, ...histories.flat());
  data.panels.forEach((panel, index) => {
    const card = cards[index];
    $('.policy-select', card).value = panel.policy;
    $('.game-frame', card).src = panel.frame;
    $('.game-frame', card).alt = `${panel.label}: ${data.mode === 'demo' ? 'entorno sintético, no Mario' : 'Mario NES real'}`;
    $('.panel-status', card).textContent = panel.status.toUpperCase();
    $('.panel-status', card).classList.toggle('failed', panel.death);
    $('.panel-status', card).classList.toggle('limited', panel.truncated);
    $('.screen-badge', card).hidden = !panel.done;
    $('.action', card).textContent = panel.action || 'Esperando primera decisión';
    $('.progress', card).textContent = number.format(panel.progress_max);
    $('.reward', card).textContent = number.format(panel.reward);
    $('.decisions', card).textContent = number.format(panel.steps);
    $('.raw-frames', card).textContent = `${number.format(panel.raw_frames)} frames`;
    const points = histories[index].map((value, i, list) =>
      `${list.length === 1 ? 0 : i / (list.length - 1) * 300},${40 - value / maxProgress * 36}`);
    if (points.length === 1) points.push(`300,${40 - panel.progress_max / maxProgress * 36}`);
    $('polyline', card).setAttribute('points', points.join(' '));
    const q = $('.q-values', card);
    q.replaceChildren();
    if (!panel.q_values) q.textContent = 'Baseline sin red neuronal: no tiene valores Q.';
    else panel.q_values.forEach((value, action) => {
      const item = document.createElement('div');
      item.className = 'q-item';
      item.classList.toggle('best', value === Math.max(...panel.q_values));
      const label = document.createElement('span');
      label.textContent = actions[action];
      const score = document.createElement('strong');
      score.textContent = value.toFixed(3);
      item.append(label, score);
      q.append(item);
    });
  });
  $('#record').textContent = data.recording ? '■ Detener y guardar' : '● Grabar MP4';
  $('#record').classList.toggle('recording', data.recording);
  $('#record-status').textContent = data.recording ? `Grabando · ${data.recording_frames} frames · máximo 1.800` : '';
  $('#video-download').hidden = !data.last_recording;
  if (data.all_done) pause();
  controls();
}

async function loop() {
  if (!playing) return;
  const start = performance.now();
  const next = await perform(() => api('/api/step', {}));
  if (next) render(next);
  if (playing) {
    const fps = 15 * Number($('#speed').value);
    timer = setTimeout(loop, Math.max(0, 1000 / fps - (performance.now() - start)));
  }
}

$('#play').addEventListener('click', () => {
  if (playing) pause();
  else if (!busy && !state?.all_done) { playing = true; controls(); loop(); }
});
$('#step').addEventListener('click', async () => {
  pause();
  const next = await perform(() => api('/api/step', {}));
  if (next) render(next);
});
$('#reset').addEventListener('click', async () => {
  pause();
  if (!$('#seed').reportValidity()) return;
  const next = await perform(() => api('/api/reset', {seed: Number($('#seed').value)}));
  if (next) render(next);
});
$('#record').addEventListener('click', async () => {
  const result = await perform(() => state.recording ? api('/api/record/stop', {}) :
    api('/api/record/start', {fps: Math.round(15 * Number($('#speed').value))}));
  if (result) {
    const next = await perform(() => api('/api/state'));
    if (next) render(next);
    if (playing) { clearTimeout(timer); timer = setTimeout(loop, 0); }
  }
});
window.addEventListener('pagehide', pause);
perform(() => api('/api/state')).then(data => { if (data) render(data); });

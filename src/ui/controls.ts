import { SCHEDULES, activationBytes, inputBytes, layersPerChunk, simulate, validateConfig } from '../sim/index.ts';
import { fmtBytes } from './format.ts';
import { t } from './i18n.ts';
import type { Key } from './i18n.ts';
import type { CommModel, ScheduleName, SimConfig } from '../sim/index.ts';
import type { Store } from './state.ts';

interface NumField {
  key: keyof SimConfig;
  label: Key;
  min: number;
  step: number;
  hint?: Key;
  /** Multiply the typed value by this factor before storing (e.g. MB -> bytes). */
  scale?: number;
  /** Fieldset the field belongs to. */
  group: 'pipeline' | 'memory';
}

const FIELDS: NumField[] = [
  { key: 'pp', label: 'pp', min: 1, step: 1, hint: 'ppHint', group: 'pipeline' },
  { key: 'vpp', label: 'vpp', min: 1, step: 1, hint: 'vppHint', group: 'pipeline' },
  { key: 'microBatches', label: 'microBatches', min: 1, step: 1, hint: 'microBatchesHint', group: 'pipeline' },
  { key: 'p2pLatency', label: 'p2pLatency', min: 0, step: 0.05, hint: 'p2pLatencyHint', group: 'pipeline' },
  { key: 'forwardTime', label: 'forwardTime', min: 0.01, step: 0.1, hint: 'forwardTimeHint', group: 'pipeline' },
  { key: 'backwardTime', label: 'backwardTime', min: 0.01, step: 0.1, hint: 'backwardTimeHint', group: 'pipeline' },
  { key: 'seqLen', label: 'seqLen', min: 1, step: 1024, group: 'memory' },
  { key: 'hiddenSize', label: 'hidden', min: 1, step: 512, group: 'memory' },
  { key: 'microBatchSize', label: 'mbs', min: 1, step: 1, hint: 'mbsHint', group: 'memory' },
  { key: 'activationMultiplier', label: 'multiplier', min: 0, step: 1, hint: 'multiplierHint', group: 'memory' },
  { key: 'numLayers', label: 'layers', min: 1, step: 1, hint: 'layersHint', group: 'memory' },
  { key: 'baselineBytes', label: 'baseline', min: 0, step: 1024, scale: 1024 ** 2, hint: 'baselineHint', group: 'memory' },
];

const SCHEDULE_DESC: Record<ScheduleName, Key> = { gpipe: 'schedGpipe', '1f1b': 'sched1f1b', 'interleaved-1f1b': 'schedInterleaved' };

const DTYPES: { label: string; bytes: number }[] = [
  { label: '1 Byte', bytes: 1 },
  { label: '2 Bytes', bytes: 2 },
  { label: '4 Bytes', bytes: 4 },
];


/** Run the simulator for the store's config and publish trace or error. */
export function recompute(store: Store): void {
  const cfg = store.get().config;
  const errors = validateConfig(cfg);
  if (errors.length) {
    store.set({ trace: null, error: errors.join('；') });
    return;
  }
  try {
    const trace = simulate(cfg);
    store.set({ trace, error: null, selectedMb: null, hoverMb: null });
  } catch (e) {
    store.set({ trace: null, error: e instanceof Error ? e.message : String(e) });
  }
}

export function mountControls(root: HTMLElement, store: Store): void {
  root.innerHTML = '';
  const form = document.createElement('form');
  form.className = 'controls';
  form.addEventListener('submit', (e) => e.preventDefault());

  const fieldset = (legend: string, layout: 'rows' | 'grid2' | 'grid3'): { fs: HTMLFieldSetElement; grid: HTMLDivElement } => {
    const fs = document.createElement('fieldset');
    fs.className = 'group';
    const lg = document.createElement('legend');
    lg.textContent = legend;
    fs.appendChild(lg);
    const grid = document.createElement('div');
    grid.className = layout;
    fs.appendChild(grid);
    form.appendChild(fs);
    return { fs, grid };
  };

  // Schedule + comm model
  const schedGroup = fieldset(t('groupSchedule'), 'rows');
  const schedWrap = document.createElement('label');
  schedWrap.className = 'field row';
  schedWrap.innerHTML = `<span>${t('schedule')}</span>`;
  const select = document.createElement('select');
  for (const info of Object.values(SCHEDULES)) {
    const o = document.createElement('option');
    o.value = info.name;
    o.textContent = info.label;
    select.appendChild(o);
  }
  schedWrap.appendChild(select);
  schedGroup.grid.appendChild(schedWrap);
  const desc = document.createElement('p');
  desc.className = 'hint span2';

  select.addEventListener('change', () => {
    const schedule = select.value as ScheduleName;
    const cfg = { ...store.get().config, schedule };
    // Defaults per schedule: GPipe / 1F1B use vpp 1, interleaved uses vpp 2.
    cfg.vpp = SCHEDULES[schedule].supportsVpp ? 2 : 1;
    store.set({ config: cfg });
    syncInputs();
    recompute(store);
  });

  // Communication model selector
  const commWrap = document.createElement('label');
  commWrap.className = 'field row';
  commWrap.innerHTML = `<span>${t('comm')}</span>`;
  const commSelect = document.createElement('select');
  const COMM_OPTIONS: { value: CommModel; label: string; hint: string }[] = [
    { value: 'async', label: t('commAsync'), hint: t('commAsyncHint') },
    { value: 'sync', label: t('commSync'), hint: t('commSyncHint') },
  ];
  for (const o of COMM_OPTIONS) {
    const opt = document.createElement('option');
    opt.value = o.value;
    opt.textContent = o.label;
    commSelect.appendChild(opt);
  }
  commWrap.appendChild(commSelect);
  schedGroup.grid.appendChild(commWrap);
  schedGroup.fs.appendChild(desc);
  const commHint = document.createElement('p');
  commHint.className = 'hint';
  schedGroup.fs.appendChild(commHint);
  commSelect.addEventListener('change', () => {
    store.set({ config: { ...store.get().config, commModel: commSelect.value as CommModel } });
    syncInputs();
    recompute(store);
  });

  // Numeric fields
  const inputs = new Map<keyof SimConfig, HTMLInputElement>();
  let dtypeSelect: HTMLSelectElement | null = null;
  const actSummary = document.createElement('p');
  actSummary.className = 'hint summary';
  const groups = { pipeline: fieldset(t('groupPipeline'), 'grid2'), memory: fieldset(t('groupMemory'), 'grid3') };
  for (const f of FIELDS) {
    const wrap = document.createElement('label');
    wrap.className = 'field';
    if (f.hint) wrap.title = t(f.hint);
    const span = document.createElement('span');
    span.textContent = t(f.label);
    wrap.appendChild(span);
    const input = document.createElement('input');
    input.type = 'number';
    input.min = String(f.min);
    input.step = String(f.step);
    input.addEventListener('input', () => {
      const v = Number(input.value);
      if (!Number.isFinite(v)) return;
      store.set({ config: { ...store.get().config, [f.key]: v * (f.scale ?? 1) } });
      syncInputs();
      recompute(store);
    });
    wrap.appendChild(input);
    groups[f.group].grid.appendChild(wrap);
    inputs.set(f.key, input);
    if (f.key === 'microBatchSize') {
      const dw = document.createElement('label');
      dw.className = 'field';
      dw.innerHTML = `<span>${t('dtype')}</span>`;
      dtypeSelect = document.createElement('select');
      for (const d of DTYPES) {
        const o = document.createElement('option');
        o.value = String(d.bytes);
        o.textContent = d.label;
        dtypeSelect.appendChild(o);
      }
      dtypeSelect.addEventListener('change', () => {
        store.set({ config: { ...store.get().config, dtypeBytes: Number(dtypeSelect!.value) } });
        syncInputs();
        recompute(store);
      });
      dw.appendChild(dtypeSelect);
      groups.memory.grid.appendChild(dw);
    }
  }
  groups.memory.fs.appendChild(actSummary);

  const errorBox = document.createElement('p');
  errorBox.className = 'error';
  form.appendChild(errorBox);
  root.appendChild(form);

  function syncInputs(): void {
    const cfg = store.get().config;
    select.value = cfg.schedule;
    desc.textContent = t(SCHEDULE_DESC[cfg.schedule]);
    const comm = cfg.commModel ?? 'async';
    commSelect.value = comm;
    commHint.textContent = COMM_OPTIONS.find((o) => o.value === comm)?.hint ?? '';
    for (const [key, input] of inputs) {
      const v = cfg[key];
      const f = FIELDS.find((x) => x.key === key)!;
      if (document.activeElement !== input) input.value = v === undefined ? '' : String((v as number) / (f.scale ?? 1));
      input.disabled = key === 'vpp' && !SCHEDULES[cfg.schedule].supportsVpp;
    }
    if (dtypeSelect) dtypeSelect.value = String(cfg.dtypeBytes ?? 2);
    const act = activationBytes(cfg);
    actSummary.textContent = t('summary', { input: fmtBytes(inputBytes(cfg)), layers: layersPerChunk(cfg), act: fmtBytes(act.input + act.intermediate) });
  }

  store.subscribe((s) => {
    errorBox.textContent = s.error ?? '';
    desc.textContent = t(SCHEDULE_DESC[s.config.schedule]);
  });
  syncInputs();
}

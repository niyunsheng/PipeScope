import { SCHEDULES, activationBytes, inputBytes, quadraticShare, simulate, validateConfig } from '../sim/index.ts';
import type { CommModel, ScheduleName, SimConfig } from '../sim/index.ts';
import { fmtBytes } from './format.ts';
import { t } from './i18n.ts';
import type { Key } from './i18n.ts';
import { generateLengths, parseCustom } from './lengths.ts';
import type { LengthMode, LengthOrder } from './lengths.ts';
import type { Store } from './state.ts';

interface NumField {
  key: keyof SimConfig;
  label: Key;
  min: number;
  step: number;
}

/** Three-column grids; order here is the visual order. */
const PIPELINE_FIELDS: NumField[] = [
  { key: 'pp', label: 'pp', min: 1, step: 1 },
  { key: 'vpp', label: 'vpp', min: 1, step: 1 },
  { key: 'microBatches', label: 'microBatches', min: 1, step: 1 },
  { key: 'p2pLatency', label: 'p2pLatency', min: 0, step: 0.05 },
  { key: 'forwardTime', label: 'forwardTime', min: 0.01, step: 0.1 },
  { key: 'backwardTime', label: 'backwardTime', min: 0.01, step: 0.1 },
];
const MODEL_FIELDS: NumField[] = [
  { key: 'seqLen', label: 'seqLen', min: 1, step: 1024 },
  { key: 'hiddenSize', label: 'hidden', min: 1, step: 512 },
  { key: 'microBatchSize', label: 'mbs', min: 1, step: 1 },
];
/** Full-width rows: input with its explanation shown beside it. */
const MODEL_ROWS: (NumField & { note: Key })[] = [
  { key: 'activationMultiplier', label: 'multiplier', min: 0, step: 1, note: 'multiplierNote' },
  { key: 'linearAttnRatio', label: 'linearAttn', min: 0, step: 1, note: 'linearAttnNote' },
];

const SCHEDULE_DESC: Record<ScheduleName, Key> = { gpipe: 'schedGpipe', '1f1b': 'sched1f1b', 'interleaved-1f1b': 'schedInterleaved' };
const COMM_OPTIONS: { value: CommModel; label: Key; hint: Key }[] = [
  { value: 'async', label: 'commAsync', hint: 'commAsyncHint' },
  { value: 'sync', label: 'commSync', hint: 'commSyncHint' },
];
const DTYPES = [1, 2, 4];

/** Regenerate `tokens` from the length settings so it always matches microBatches / seqLen. */
export function syncTokens(cfg: SimConfig, custom: number[]): SimConfig {
  const mode: LengthMode = cfg.lengthMode ?? 'uniform';
  if (mode === 'uniform') {
    const { tokens: _t, ...rest } = cfg;
    return rest;
  }
  const tokens = generateLengths({
    n: cfg.microBatches,
    mean: cfg.seqLen ?? 4096,
    mode,
    cv: cfg.lengthCv ?? 0,
    seed: cfg.lengthSeed ?? 1,
    order: cfg.lengthOrder ?? 'asis',
    custom,
  });
  return { ...cfg, tokens };
}

/** Custom token list typed by the user (kept outside SimConfig; cycled into `tokens`). */
let customTokens: number[] = [];

/** Run the simulator for the store's config and publish trace or error. */
export function recompute(store: Store): void {
  const synced = syncTokens(store.get().config, customTokens);
  store.set({ config: synced });
  const errors = validateConfig(synced);
  if (errors.length) {
    store.set({ trace: null, error: errors.join('; ') });
    return;
  }
  try {
    const trace = simulate(synced);
    store.set({ trace, error: null, selectedMb: null, hoverMb: null });
  } catch (e) {
    store.set({ trace: null, error: e instanceof Error ? e.message : String(e) });
  }
}

// ---- small DOM helpers -----------------------------------------------------

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

function fieldset(parent: HTMLElement, legend: string): { fs: HTMLFieldSetElement; grid: HTMLDivElement } {
  const fs = el('fieldset', 'group');
  fs.appendChild(el('legend', undefined, legend));
  const grid = el('div', 'grid3');
  fs.appendChild(grid);
  parent.appendChild(fs);
  return { fs, grid };
}

function select<V extends string>(options: { value: V; label: string }[], onChange: (v: V) => void): HTMLSelectElement {
  const sel = el('select');
  for (const o of options) {
    const opt = el('option', undefined, o.label);
    opt.value = o.value;
    sel.appendChild(opt);
  }
  sel.addEventListener('change', () => onChange(sel.value as V));
  return sel;
}

function numberInput(min: number, step: number, onChange: (v: number) => void): HTMLInputElement {
  const input = el('input');
  input.type = 'number';
  input.min = String(min);
  input.step = String(step);
  input.addEventListener('input', () => {
    const v = Number(input.value);
    if (Number.isFinite(v)) onChange(v);
  });
  return input;
}

/** Label above input (grid cell). */
function cell(label: string, control: HTMLElement): HTMLLabelElement {
  const wrap = el('label', 'field');
  wrap.appendChild(el('span', undefined, label));
  wrap.appendChild(control);
  return wrap;
}

/** Full-width row: label | control | note (note wraps to the available width). */
function row(label: string, control: HTMLElement, note: HTMLElement): HTMLLabelElement {
  const wrap = el('label', 'field row-note span3');
  wrap.appendChild(el('span', undefined, label));
  wrap.appendChild(control);
  wrap.appendChild(note);
  return wrap;
}

// ---- panel -----------------------------------------------------------------

/**
 * Mount the parameter panel into `root` and the schedule selector into
 * `scheduleSlot` (the top bar). Explanations are rendered next to the inputs.
 */
export function mountControls(root: HTMLElement, scheduleSlot: HTMLElement, store: Store): void {
  root.innerHTML = '';
  const form = el('form', 'controls');
  form.addEventListener('submit', (e) => e.preventDefault());
  root.appendChild(form);

  const update = (patch: Partial<SimConfig>): void => {
    store.set({ config: { ...store.get().config, ...patch } });
    recompute(store);
  };

  // Schedule selector lives in the top bar as a segmented control.
  const schedButtons = new Map<ScheduleName, HTMLButtonElement>();
  for (const info of Object.values(SCHEDULES)) {
    const b = el('button', undefined, info.label);
    b.type = 'button';
    b.addEventListener('click', () => update({ schedule: info.name, vpp: info.supportsVpp ? 2 : 1 }));
    schedButtons.set(info.name, b);
    scheduleSlot.appendChild(b);
  }
  const schedDesc = el('p', 'note');

  // Pipeline
  const pipe = fieldset(form, t('groupPipeline'));
  const inputs = new Map<keyof SimConfig, HTMLInputElement>();
  const addNum = (grid: HTMLElement, f: NumField) => {
    const input = numberInput(f.min, f.step, (v) => update({ [f.key]: v }));
    inputs.set(f.key, input);
    grid.appendChild(cell(t(f.label), input));
  };
  pipe.fs.insertBefore(schedDesc, pipe.grid);
  for (const f of PIPELINE_FIELDS) addNum(pipe.grid, f);
  const commSelect = select(
    COMM_OPTIONS.map((o) => ({ value: o.value, label: t(o.label) })),
    (v: CommModel) => update({ commModel: v }),
  );
  const commNote = el('p', 'note');
  pipe.grid.appendChild(row(t('comm'), commSelect, commNote));

  // Model
  const model = fieldset(form, t('groupMemory'));
  for (const f of MODEL_FIELDS) addNum(model.grid, f);
  const dtypeSelect = select(
    DTYPES.map((b) => ({ value: String(b), label: `${b} Byte${b > 1 ? 's' : ''}` })),
    (v: string) => update({ dtypeBytes: Number(v) }),
  );
  model.grid.appendChild(cell(t('dtype'), dtypeSelect));
  addNum(model.grid, { key: 'layersPerChunk', label: 'layers', min: 0, step: 1 });
  const modelNote = el('p', 'note span3');
  model.grid.appendChild(modelNote);
  const rowNotes = new Map<keyof SimConfig, HTMLParagraphElement>();
  for (const f of MODEL_ROWS) {
    const input = numberInput(f.min, f.step, (v) => update({ [f.key]: v }));
    inputs.set(f.key, input);
    const note = el('p', 'note');
    rowNotes.set(f.key, note);
    model.grid.appendChild(row(t(f.label), input, note));
  }

  // Micro-batch lengths
  const len = fieldset(form, t('groupLengths'));
  const modeSelect = select<LengthMode>(
    [
      { value: 'uniform', label: t('modeUniform') },
      { value: 'lognormal', label: t('modeLognormal') },
      { value: 'custom', label: t('modeCustom') },
    ],
    (v) => update({ lengthMode: v }),
  );
  const modeCell = cell(t('lengthMode'), modeSelect);
  modeCell.classList.add('span2');
  len.grid.appendChild(modeCell);
  const cvInput = numberInput(0, 0.01, (v) => update({ lengthCv: v }));
  const cvCell = cell(t('cv'), cvInput);
  len.grid.appendChild(cvCell);
  const seedInput = numberInput(0, 1, (v) => update({ lengthSeed: v }));
  const seedCell = cell(t('seed'), seedInput);
  len.grid.appendChild(seedCell);
  const orderSelect = select<LengthOrder>(
    [
      { value: 'asis', label: t('orderAsis') },
      { value: 'asc', label: t('orderAsc') },
      { value: 'desc', label: t('orderDesc') },
      { value: 'alternate', label: t('orderAlternate') },
    ],
    (v) => update({ lengthOrder: v }),
  );
  const orderCell = cell(t('order'), orderSelect);
  orderCell.classList.add('span2');
  len.grid.appendChild(orderCell);
  const customInput = el('textarea');
  customInput.rows = 2;
  customInput.placeholder = '4096, 2048, 8192, ...';
  customInput.addEventListener('input', () => {
    customTokens = parseCustom(customInput.value);
    recompute(store);
  });
  const customCell = cell(t('customLabel'), customInput);
  customCell.classList.add('span3');
  len.grid.appendChild(customCell);
  const lenNote = el('p', 'note span3');
  len.grid.appendChild(lenNote);

  const errorBox = el('p', 'error');
  form.appendChild(errorBox);

  if ((store.get().config.lengthMode ?? 'uniform') === 'custom' && store.get().config.tokens) {
    customTokens = store.get().config.tokens!.slice();
    customInput.value = customTokens.join(', ');
  }

  const setValue = (input: HTMLInputElement, v: number | undefined) => {
    if (document.activeElement !== input) input.value = v === undefined ? '' : String(v);
  };

  store.subscribe((s) => {
    const cfg = s.config;
    errorBox.textContent = s.error ?? '';
    for (const [name, b] of schedButtons) b.classList.toggle('active', name === cfg.schedule);
    schedDesc.textContent = t(SCHEDULE_DESC[cfg.schedule]);
    for (const [key, input] of inputs) {
      setValue(input, cfg[key] as number | undefined);
      input.disabled = key === 'vpp' && !SCHEDULES[cfg.schedule].supportsVpp;
    }
    const comm = cfg.commModel ?? 'async';
    commSelect.value = comm;
    commNote.textContent = t(COMM_OPTIONS.find((o) => o.value === comm)!.hint);
    dtypeSelect.value = String(cfg.dtypeBytes ?? 2);
    const act = activationBytes(cfg);
    modelNote.textContent = `${t('noteTimeUnit')} ${t('noteInput', { input: fmtBytes(inputBytes(cfg)), act: fmtBytes(act.input + act.intermediate), layers: cfg.layersPerChunk ?? 2, mult: cfg.activationMultiplier ?? 17 })}`;
    rowNotes.get('activationMultiplier')!.textContent = t('multiplierNote');
    rowNotes.get('linearAttnRatio')!.textContent = t('linearAttnNote', { alpha: quadraticShare(cfg).toFixed(3) });
    const mode = cfg.lengthMode ?? 'uniform';
    modeSelect.value = mode;
    orderSelect.value = cfg.lengthOrder ?? 'asis';
    setValue(cvInput, cfg.lengthCv ?? 0);
    setValue(seedInput, cfg.lengthSeed ?? 1);
    cvCell.hidden = mode !== 'lognormal';
    seedCell.hidden = mode !== 'lognormal';
    customCell.hidden = mode !== 'custom';
    orderCell.hidden = mode === 'uniform';
    lenNote.hidden = mode === 'uniform';
    lenNote.textContent = mode === 'custom' ? t('customHint') : t('noteLengths', { alpha: quadraticShare(cfg).toFixed(3) });
  });
}

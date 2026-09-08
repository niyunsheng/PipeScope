import { DEFAULT_CONFIG, SCHEDULES, activationBytes, builtinWarmupFormula, inputBytes, layersPerChunk, megatronPlacement, parseFormula, quadraticShare, simulate, validateConfig, warmupVars } from '../sim/index.ts';
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
/** Rows of three: pipeline shape / compute costs / batch and network. */
const PIPELINE_FIELDS: NumField[] = [
  { key: 'pp', label: 'pp', min: 1, step: 1 },
  { key: 'vpp', label: 'vpp', min: 1, step: 1 },
  { key: 'groupSize', label: 'groupSize', min: 1, step: 1 },
  { key: 'forwardTime', label: 'forwardTime', min: 0.01, step: 0.1 },
  { key: 'backwardTime', label: 'backwardTime', min: 0.01, step: 0.1 },
  { key: 'lossTime', label: 'lossTime', min: 0, step: 0.1 },
  { key: 'microBatches', label: 'microBatches', min: 1, step: 1 },
  { key: 'p2pLatency', label: 'p2pLatency', min: 0, step: 0.05 },
];
const MODEL_FIELDS: NumField[] = [
  { key: 'seqLen', label: 'seqLen', min: 1, step: 1024 },
  { key: 'hiddenSize', label: 'hidden', min: 1, step: 512 },
];
/** Full-width rows: input with its explanation shown beside it. */
const MODEL_ROWS: (NumField & { note: Key })[] = [
  { key: 'activationMultiplier', label: 'multiplier', min: 0, step: 1, note: 'multiplierNote' },
  { key: 'linearAttnRatio', label: 'linearAttn', min: 0, step: 1, note: 'linearAttnNote' },
];

const SCHEDULE_DESC: Record<ScheduleName, Key> = { gpipe: 'schedGpipe', '1f1b': 'sched1f1b', 'interleaved-1f1b': 'schedInterleaved', custom: 'schedCustom' };
const COMM_OPTIONS: { value: CommModel; label: Key; hint: Key }[] = [
  { value: 'async', label: 'commAsync', hint: 'commAsyncHint' },
  { value: 'sync', label: 'commSync', hint: 'commSyncHint' },
];
const DTYPES = [1, 2, 4];

/** Regenerate `tokens` from the length settings so it always matches microBatches / seqLen. */
export function syncTokens(cfg: SimConfig, custom: number[]): SimConfig {
  const mode: LengthMode = cfg.lengthMode;
  if (mode === 'uniform') {
    const { tokens: _t, ...rest } = cfg;
    return rest;
  }
  const tokens = generateLengths({
    n: cfg.microBatches,
    mean: cfg.seqLen,
    mode,
    cv: cfg.lengthCv,
    seed: cfg.lengthSeed,
    order: cfg.lengthOrder,
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
    // A failed run still carries everything that ran; the timeline marks the failure in red.
    store.set({ trace, error: trace.failure ? trace.failure.message : null, selectedMb: null, selectedOp: null, selectedTransfer: null });
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

/**
 * Segmented button group for a small set of mutually exclusive modes, styled
 * like the schedule selector in the top bar. `set` highlights the active value.
 */
function segmented<V extends string>(
  options: { value: V; label: string }[],
  onChange: (v: V) => void,
): { root: HTMLDivElement; set: (v: V) => void } {
  const root = el('div', 'segmented');
  const buttons = new Map<V, HTMLButtonElement>();
  for (const o of options) {
    const b = el('button', undefined, o.label);
    b.type = 'button';
    b.addEventListener('click', () => onChange(o.value));
    buttons.set(o.value, b);
    root.appendChild(b);
  }
  const set = (v: V) => {
    for (const [value, b] of buttons) b.classList.toggle('active', value === v);
  };
  const enable = (on: boolean) => {
    for (const b of buttons.values()) b.disabled = !on;
    root.classList.toggle('disabled', !on);
  };
  return { root, set, enable };
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
    const prev = store.get().config;
    const next = { ...prev, ...patch };
    // Group size follows pp (Megatron's default) until the user sets it to something else.
    if (patch.pp !== undefined && patch.groupSize === undefined && prev.groupSize === prev.pp) next.groupSize = patch.pp;
    store.set({ config: next });
    recompute(store);
  };

  // Schedule selector lives in the top bar as a segmented control.
  const schedButtons = new Map<ScheduleName, HTMLButtonElement>();
  for (const info of Object.values(SCHEDULES)) {
    const b = el('button', undefined, info.label);
    b.type = 'button';
    // Switching schedule also resets the placement to what Megatron uses for it.
    b.addEventListener('click', () => {
      // 1F1B (and the GPipe baseline, which mirrors its communication) are blocking-only.
      const comm = info.name === '1f1b' || info.name === 'gpipe' ? 'sync' : store.get().config.commModel;
      // Switching to a blocking-only schedule drops the settings that only exist on the isend / irecv path.
      const prefetch = comm === 'async' && store.get().config.prefetchWarmupFlush;
      update({ schedule: info.name, vpp: info.supportsVpp ? 2 : 1, commModel: comm, prefetchWarmupFlush: prefetch, ...megatronPlacement(info.name, comm) });
    });
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
    if (f.key !== 'p2pLatency') {
      grid.appendChild(cell(t(f.label), input));
      return;
    }
    // Latency gets a one-click toggle between 0 (pure schedule shape, comm
    // rows hidden) and the last non-zero value.
    let lastNonZero = DEFAULT_CONFIG.p2pLatency;
    const zero = el('button', 'mini-btn');
    zero.type = 'button';
    zero.addEventListener('click', () => {
      const cur = store.get().config.p2pLatency;
      if (cur > 0) {
        lastNonZero = cur;
        update({ p2pLatency: 0 });
      } else {
        update({ p2pLatency: lastNonZero });
      }
    });
    latencyToggle = zero;
    const wrap = el('div', 'with-btn');
    wrap.appendChild(input);
    wrap.appendChild(zero);
    grid.appendChild(cell(t(f.label), wrap));
  };
  let latencyToggle: HTMLButtonElement | null = null;
  pipe.fs.insertBefore(schedDesc, pipe.grid);
  for (const f of PIPELINE_FIELDS) addNum(pipe.grid, f);
  const commSeg = segmented(
    COMM_OPTIONS.map((o) => ({ value: o.value, label: t(o.label) })),
    (v: CommModel) => {
      // For the built-in schedules the comm model also picks Megatron's path (and its placement).
      const sched = store.get().config.schedule;
      update({ commModel: v, ...(v === 'async' ? {} : { prefetchWarmupFlush: false }), ...(sched === 'custom' ? {} : megatronPlacement(sched, v)) });
    },
  );
  const commNote = el('p', 'note');
  pipe.grid.appendChild(row(t('comm'), commSeg.root, commNote));
  // Where communication sits in the steady state: two independent switches,
  // each laid out like the comm-model row (label | control | note).
  const sendSeg = segmented<'F' | 'B'>(
    [
      { value: 'F', label: t('sendAfterF') },
      { value: 'B', label: t('sendAfterB') },
    ],
    (v) => update({ sendAfter: v }),
  );
  const sendNote = el('p', 'note');
  const sendCell = row(t('sendAfter'), sendSeg.root, sendNote);
  pipe.grid.appendChild(sendCell);
  const waitSeg = segmented<'beforeF' | 'beforeB'>(
    [
      { value: 'beforeF', label: t('waitBeforeF') },
      { value: 'beforeB', label: t('waitBeforeB') },
    ],
    (v) => update({ waitGrad: v }),
  );
  const waitNote = el('p', 'note');
  const waitCell = row(t('waitGrad'), waitSeg.root, waitNote);
  pipe.grid.appendChild(waitCell);
  // Megatron path C: warmup / cooldown prefetch on top of isend / irecv.
  const prefetchSeg = segmented<'off' | 'on'>(
    [
      { value: 'off', label: t('prefetchOff') },
      { value: 'on', label: t('prefetchOn') },
    ],
    (v) => update({ prefetchWarmupFlush: v === 'on' }),
  );
  const prefetchNote = el('p', 'note');
  const prefetchCell = row(t('prefetch'), prefetchSeg.root, prefetchNote);
  pipe.grid.appendChild(prefetchCell);

  // Warmup formula: read-only for the built-in schedules (shows what they
  // use), editable for `custom`. Same row layout as the switches above.
  const formulaInput = el('input');
  formulaInput.type = 'text';
  formulaInput.spellcheck = false;
  formulaInput.addEventListener('input', () => update({ warmupFormula: formulaInput.value }));
  const formulaNote = el('p', 'note');
  const formulaCell = row(t('warmupFormula'), formulaInput, formulaNote);
  formulaCell.classList.add('formula-row');
  pipe.grid.appendChild(formulaCell);

  // Model
  const model = fieldset(form, t('groupMemory'));
  model.grid.className = 'grid4'; // the four shape fields on one row (micro-batch size stays 1)
  for (const f of MODEL_FIELDS) addNum(model.grid, f);
  const dtypeSelect = select(
    DTYPES.map((b) => ({ value: String(b), label: `${b} Byte${b > 1 ? 's' : ''}` })),
    (v: string) => update({ dtypeBytes: Number(v) }),
  );
  model.grid.appendChild(cell(t('dtype'), dtypeSelect));
  addNum(model.grid, { key: 'numLayers', label: 'layers', min: 1, step: 1 });
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
  const modeSeg = segmented<LengthMode>(
    [
      { value: 'uniform', label: t('modeUniform') },
      { value: 'lognormal', label: t('modeLognormal') },
      { value: 'custom', label: t('modeCustom') },
    ],
    (v) => {
      // Entering custom mode with an empty list: seed it from the lengths
      // currently in effect (uniform seqLen x microBatches by default) so the
      // textarea always shows an editable starting point.
      if (v === 'custom' && customTokens.length === 0) {
        const cfg = store.get().config;
        customTokens = cfg.tokens?.slice() ?? Array.from({ length: cfg.microBatches }, () => cfg.seqLen);
      }
      update({ lengthMode: v });
    },
  );
  const modeCell = cell(t('lengthMode'), modeSeg.root);
  modeCell.classList.add('span3');
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

  if (store.get().config.lengthMode === 'custom' && store.get().config.tokens) {
    customTokens = store.get().config.tokens!.slice();
    customInput.value = customTokens.join(', ');
  }

  const setValue = (input: HTMLInputElement, v: number | undefined) => {
    if (document.activeElement !== input) input.value = v === undefined ? '' : String(v);
  };

  store.subscribeTo(['config', 'error'], (s) => {
    const cfg = s.config;
    errorBox.textContent = s.error ?? '';
    for (const [name, b] of schedButtons) b.classList.toggle('active', name === cfg.schedule);
    schedDesc.textContent = t(SCHEDULE_DESC[cfg.schedule]);
    for (const [key, input] of inputs) {
      setValue(input, cfg[key] as number | undefined);
      input.disabled = (key === 'vpp' || key === 'groupSize') && !SCHEDULES[cfg.schedule].supportsVpp;
    }
    // Warmup formula row: built-in schedules show theirs greyed out; custom edits.
    const editable = cfg.schedule === 'custom';
    const formula = editable ? cfg.warmupFormula : builtinWarmupFormula(cfg.schedule);
    formulaInput.disabled = !editable;
    if (document.activeElement !== formulaInput) formulaInput.value = formula;
    try {
      const f = parseFormula(formula);
      const total = cfg.microBatches * cfg.vpp;
      const vals = Array.from({ length: cfg.pp }, (_, r) => Math.max(0, Math.min(total, Math.round(f.eval(warmupVars(cfg, r))))));
      // Steady 1F1B steps per rank = total - warmup (cooldown mirrors warmup).
      const lines = [t('warmupValues', { v: vals.join(', ') }), t('steadyValues', { v: vals.map((w) => total - w).join(', ') })];
      if (editable) lines.push(t('warmupVarsHint'));
      formulaNote.textContent = lines.join('\n');
      formulaNote.classList.remove('error');
    } catch (e) {
      formulaNote.textContent = e instanceof Error ? e.message : String(e);
      formulaNote.classList.add('error');
    }
    // Placement switches: irrelevant for GPipe (no steady state); read-only
    // for the built-in schedules, where they show Megatron's placement.
    sendSeg.set(cfg.sendAfter);
    waitSeg.set(cfg.waitGrad);
    const noSteady = cfg.schedule === 'gpipe';
    sendCell.hidden = noSteady;
    waitCell.hidden = noSteady;
    sendSeg.enable(cfg.schedule === 'custom');
    waitSeg.enable(cfg.schedule === 'custom');
    sendNote.textContent = t(cfg.sendAfter === 'F' ? 'sendAfterFHint' : 'sendAfterBHint');
    waitNote.textContent = t(cfg.waitGrad === 'beforeF' ? 'waitBeforeFHint' : 'waitBeforeBHint');
    // Prefetch exists only on the isend / irecv path of the interleaved skeleton;
    // shown greyed out elsewhere, like the comm-model row.
    const hasComm = cfg.schedule === 'interleaved-1f1b' || cfg.schedule === 'custom';
    prefetchSeg.set(cfg.prefetchWarmupFlush ? 'on' : 'off');
    prefetchSeg.enable(hasComm && cfg.commModel === 'async');
    prefetchNote.textContent = t(cfg.prefetchWarmupFlush ? 'prefetchOnHint' : 'prefetchOffHint');
    if (latencyToggle) {
      latencyToggle.textContent = cfg.p2pLatency > 0 ? t('latencyZero') : t('latencyRestore');
      latencyToggle.title = cfg.p2pLatency > 0 ? t('latencyZeroHint') : t('latencyRestoreHint');
    }
    const comm = cfg.commModel;
    commSeg.set(comm);
    commSeg.enable(cfg.schedule === 'interleaved-1f1b' || cfg.schedule === 'custom');
    // Hint: the transport semantics, plus which Megatron path this selects for the current schedule.
    const pathHint =
      cfg.schedule === 'interleaved-1f1b' ? t(comm === 'async' ? 'commPathVppAsync' : 'commPathVppSync')
      : cfg.schedule === '1f1b' ? t('commPath1f1bSync')
      : cfg.schedule === 'gpipe' ? t('commPathGpipe')
      : '';
    commNote.textContent = `${t(COMM_OPTIONS.find((o) => o.value === comm)!.hint)} ${pathHint}`.trim();
    dtypeSelect.value = String(cfg.dtypeBytes);
    const act = activationBytes(cfg);
    modelNote.textContent = `${t('noteTimeUnit', { lpc: layersPerChunk(cfg), stages: cfg.pp * cfg.vpp })} ${t('noteInput', { input: fmtBytes(inputBytes(cfg)), act: fmtBytes(act.input + act.intermediate), layers: layersPerChunk(cfg), mult: cfg.activationMultiplier })}`;
    rowNotes.get('activationMultiplier')!.textContent = t('multiplierNote');
    rowNotes.get('linearAttnRatio')!.textContent = t('linearAttnNote', { alpha: quadraticShare(cfg).toFixed(3) });
    const mode = cfg.lengthMode;
    modeSeg.set(mode);
    orderSelect.value = cfg.lengthOrder;
    setValue(cvInput, cfg.lengthCv);
    setValue(seedInput, cfg.lengthSeed);
    cvCell.hidden = mode !== 'lognormal';
    seedCell.hidden = mode !== 'lognormal';
    customCell.hidden = mode !== 'custom';
    if (mode === 'custom' && document.activeElement !== customInput) {
      customInput.value = customTokens.join(', ');
    }
    orderCell.hidden = mode === 'uniform';
    lenNote.hidden = mode === 'uniform';
    lenNote.textContent = mode === 'custom' ? t('customHint') : t('noteLengths', { alpha: quadraticShare(cfg).toFixed(3) });
  });
}

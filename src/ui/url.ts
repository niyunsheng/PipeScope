import type { CommModel, ScheduleName, SimConfig } from '../sim/index.ts';

/**
 * Shareable URLs: the configuration and the pinned micro-batch are mirrored
 * into the query string (`?schedule=1f1b&pp=4&vpp=1&mb=8&tf=1&tb=2&p2p=0&sel=3`).
 */
const KEYS: [keyof SimConfig, string][] = [
  ['pp', 'pp'],
  ['vpp', 'vpp'],
  ['microBatches', 'mb'],
  ['forwardTime', 'tf'],
  ['backwardTime', 'tb'],
  ['p2pLatency', 'p2p'],
  ['seqLen', 's'],
  ['hiddenSize', 'h'],
  ['microBatchSize', 'b'],
  ['dtypeBytes', 'dtype'],
  ['activationMultiplier', 'mult'],
  ['numLayers', 'layers'],
  ['baselineBytes', 'base'],
];

export function readUrl(defaults: SimConfig): { config: SimConfig; selectedMb: number | null } {
  const q = new URLSearchParams(location.search);
  const config: SimConfig = { ...defaults };
  const schedule = q.get('schedule');
  if (schedule) config.schedule = schedule as ScheduleName;
  const comm = q.get('comm');
  if (comm === 'async' || comm === 'sync') config.commModel = comm as CommModel;
  for (const [key, name] of KEYS) {
    const v = q.get(name);
    if (v !== null && Number.isFinite(Number(v))) (config as unknown as Record<string, number>)[key] = Number(v);
  }
  const sel = q.get('sel');
  return { config, selectedMb: sel !== null && Number.isFinite(Number(sel)) ? Number(sel) : null };
}

export function writeUrl(config: SimConfig, selectedMb: number | null): void {
  const q = new URLSearchParams();
  q.set('schedule', config.schedule);
  q.set('comm', config.commModel ?? 'async');
  for (const [key, name] of KEYS) {
    const v = config[key];
    if (v !== undefined) q.set(name, String(v));
  }
  if (selectedMb !== null) q.set('sel', String(selectedMb));
  history.replaceState(null, '', `?${q.toString()}`);
}

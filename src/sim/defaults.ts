import type { CommModel, ScheduleName, SimConfig } from './types.ts';

/**
 * Warmup formula each built-in schedule uses, in the same language as the
 * custom schedule's editable one (see `sim/formula.ts`). Shown read-only.
 */
export function builtinWarmupFormula(schedule: ScheduleName): string {
  switch (schedule) {
    case 'gpipe':
      return 'total';
    case '1f1b':
      return 'pp - r - 1';
    default:
      return '2 * (pp - r - 1) + (vpp - 1) * G';
  }
}

/**
 * Communication placement Megatron uses for a schedule under a comm model.
 *  - 1F1B: send right after F, wait right before B, and always blocking
 *    (Megatron has no non-blocking 1F1B; the UI locks the comm model to sync).
 *  - interleaved: `sync` selects the synchronous path (one batched call after
 *    B, wait before F); `async` selects the overlap path (`overlap_p2p_comm =
 *    True`: isend after F, wait right before B). These are the only two
 *    combinations Megatron implements; `custom` may use any.
 *  - GPipe has no steady state, so the placement is moot.
 */
export function megatronPlacement(schedule: ScheduleName, commModel: CommModel): Pick<SimConfig, 'sendAfter' | 'waitGrad'> {
  if (schedule === 'interleaved-1f1b' || schedule === 'custom') {
    return commModel === 'sync' ? { sendAfter: 'B', waitGrad: 'beforeF' } : { sendAfter: 'F', waitGrad: 'beforeB' };
  }
  return { sendAfter: 'F', waitGrad: 'beforeB' };
}

/** The configuration shown on first load; also the base that tests build on. */
export const DEFAULT_CONFIG: SimConfig = {
  schedule: 'interleaved-1f1b',
  pp: 4,
  vpp: 2,
  microBatches: 8,
  forwardTime: 1,
  backwardTime: 2,
  lossTime: 0,
  p2pLatency: 0.2,
  commModel: 'async',
  warmupFormula: '2 * (pp - r - 1) + (vpp - 1) * G',
  groupSize: 4,
  sendAfter: 'F',
  waitGrad: 'beforeB',
  seqLen: 4096,
  hiddenSize: 4096,
  microBatchSize: 1,
  dtypeBytes: 2,
  activationMultiplier: 17,
  layersPerChunk: 2,
  linearAttnRatio: 6,
  baselineBytes: 0,
  lengthMode: 'uniform',
  lengthCv: 0.05,
  lengthSeed: 1,
  lengthOrder: 'asis',
};

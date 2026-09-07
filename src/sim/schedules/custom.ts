import type { Program, SimConfig } from '../types.ts';
import { parseFormula } from '../formula.ts';
import type { Vars } from '../formula.ts';
import { interleavedFamily } from './interleaved.ts';

/** Variables a warmup formula may use; documented in the UI next to the field. */
export function warmupVars(cfg: SimConfig, rank: number): Vars {
  return { pp: cfg.pp, vpp: cfg.vpp, m: cfg.microBatches, r: rank, G: cfg.groupSize, total: cfg.microBatches * cfg.vpp };
}

/**
 * The interleaved skeleton with user-controlled warmup formula, group size
 * and send timing. With the defaults it reproduces Megatron exactly; the
 * point is to change one thing and watch the timeline (or the failure).
 */
export function customProgram(cfg: SimConfig): Program {
  const formula = parseFormula(cfg.warmupFormula);
  return interleavedFamily(cfg, {
    groupSize: cfg.groupSize,
    warmupOf: (rank) => formula.eval(warmupVars(cfg, rank)),
    sendAfter: cfg.sendAfter,
    waitGrad: cfg.waitGrad,
    nonblocking: cfg.commModel === 'async',
  });
}

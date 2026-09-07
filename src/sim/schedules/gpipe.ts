import type { Program, SimConfig } from '../types.ts';
import { interleavedFamily } from './interleaved.ts';

/**
 * GPipe-style schedule: all forwards, then all backwards. Not a Megatron
 * schedule; it is the textbook baseline that makes the 1F1B memory advantage
 * visible. It is the all-warmup instance of the interleaved skeleton, so its
 * communication is paired exactly like 1F1B's (each send goes out together
 * with the next recv) and the two schedules differ only in op order, i.e. in
 * activation memory.
 */
export function gpipeProgram(cfg: SimConfig): Program {
  if (cfg.vpp !== 1) throw new Error('GPipe schedule requires vpp = 1');
  return interleavedFamily(cfg, {
    groupSize: cfg.microBatches,
    warmupOf: () => cfg.microBatches, // every forward is warmup; the steady loop is empty
    sendAfter: 'F',
    waitGrad: 'beforeB',
    nonblocking: false,
  });
}

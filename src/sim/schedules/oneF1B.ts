import type { Program, SimConfig } from '../types.ts';
import { interleavedFamily } from './interleaved.ts';

/**
 * Megatron-LM non-interleaved 1F1B
 * (`forward_backward_pipelining_without_interleaving`, core_v0.19.0 L2120):
 * warmup = pp - rank - 1 forwards, then F / B alternation, then the backwards.
 * It is the single-chunk instance of the interleaved skeleton; Megatron's own
 * placement is `sendAfter = 'F'` (send_forward_recv_backward right after the
 * forward) and `waitGrad = 'beforeB'`. The other placements are exposed so
 * the effect of moving a send or a wait can be seen on this simpler schedule.
 */
export function oneF1BProgram(cfg: SimConfig): Program {
  if (cfg.vpp !== 1) throw new Error('1F1B schedule requires vpp = 1 (use interleaved-1f1b for vpp > 1)');
  return interleavedFamily(cfg, {
    groupSize: cfg.microBatches, // one chunk: the group structure is irrelevant
    warmupOf: (rank) => cfg.pp - rank - 1,
    sendAfter: cfg.sendAfter,
    waitGrad: cfg.waitGrad,
    nonblocking: cfg.commModel === 'async',
  });
}

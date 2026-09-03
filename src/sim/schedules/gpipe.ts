import type { Program, SimConfig, Step } from '../types.ts';
import { Topology, comm, compute, push } from './common.ts';

/**
 * GPipe-style schedule: all forwards, then all backwards, with a blocking
 * point-to-point exchange around every op. Not a Megatron schedule; it is the
 * textbook baseline that makes the 1F1B memory advantage visible.
 */
export function gpipeProgram(cfg: SimConfig): Program {
  if (cfg.vpp !== 1) throw new Error('GPipe schedule requires vpp = 1');
  const topo = new Topology(cfg.pp, 1);
  const program: Program = [];
  for (let rank = 0; rank < cfg.pp; rank++) {
    const steps: Step[] = [];
    for (let mb = 0; mb < cfg.microBatches; mb++) {
      push(steps, comm([], [topo.recvF(rank, mb, 0)]));
      steps.push(compute('F', mb, 0));
      push(steps, comm([topo.sendF(rank, mb, 0)], []));
    }
    for (let mb = 0; mb < cfg.microBatches; mb++) {
      push(steps, comm([], [topo.recvB(rank, mb, 0)]));
      steps.push(compute('B', mb, 0));
      push(steps, comm([topo.sendB(rank, mb, 0)], []));
    }
    program.push(steps);
  }
  return program;
}

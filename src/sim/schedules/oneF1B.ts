import type { Program, SimConfig, Step } from '../types.ts';
import { Topology, comm, compute, push } from './common.ts';

/**
 * Megatron-LM non-interleaved 1F1B
 * (`forward_backward_pipelining_without_interleaving`).
 *
 * Per rank:
 *   warmup   = min(pp - rank - 1, m) forwards, each: recv_forward, F, send_forward
 *   steady   = m - warmup iterations of: F, send_forward_recv_backward, B,
 *              send_backward_recv_forward (send_backward only on the last one)
 *   cooldown = warmup backwards, each: recv_backward, B, send_backward
 */
export function oneF1BProgram(cfg: SimConfig): Program {
  if (cfg.vpp !== 1) throw new Error('1F1B schedule requires vpp = 1 (use interleaved-1f1b for vpp > 1)');
  const topo = new Topology(cfg.pp, 1);
  const m = cfg.microBatches;
  const program: Program = [];
  for (let rank = 0; rank < cfg.pp; rank++) {
    const steps: Step[] = [];
    const warmup = Math.min(cfg.pp - rank - 1, m);
    const remaining = m - warmup;
    let fwd = 0;
    let bwd = 0;

    for (let i = 0; i < warmup; i++) {
      push(steps, comm([], [topo.recvF(rank, fwd, 0)]));
      steps.push(compute('F', fwd, 0));
      push(steps, comm([topo.sendF(rank, fwd, 0)], []));
      fwd++;
    }
    if (remaining > 0) push(steps, comm([], [topo.recvF(rank, fwd, 0)]));
    for (let i = 0; i < remaining; i++) {
      const last = i === remaining - 1;
      steps.push(compute('F', fwd, 0));
      push(steps, comm([topo.sendF(rank, fwd, 0)], [topo.recvB(rank, bwd, 0)]));
      steps.push(compute('B', bwd, 0));
      if (last) {
        push(steps, comm([topo.sendB(rank, bwd, 0)], []));
      } else {
        push(steps, comm([topo.sendB(rank, bwd, 0)], [topo.recvF(rank, fwd + 1, 0)]));
      }
      fwd++;
      bwd++;
    }
    for (let i = 0; i < warmup; i++) {
      push(steps, comm([], [topo.recvB(rank, bwd, 0)]));
      steps.push(compute('B', bwd, 0));
      push(steps, comm([topo.sendB(rank, bwd, 0)], []));
      bwd++;
    }
    program.push(steps);
  }
  return program;
}

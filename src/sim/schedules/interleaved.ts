import type { Program, SimConfig, Step } from '../types.ts';
import { Topology, comm, compute, push } from './common.ts';

/**
 * Schedule lookup table, copied from Megatron-LM `get_schedule_table`.
 * Maps a virtual micro-batch id to (micro-batch id, model chunk id).
 * For PP2 / m=5 / vpp=2 with group size 2 (last group is partial):
 *   virtual id | 0 1 2 3 4 5 6 7 8 9
 *   micro-batch| 0 1 0 1 2 3 4 2 3 4   <- wait: groups are [0,1] and [2,3,4]
 *   chunk      | 0 0 1 1 0 0 0 1 1 1
 */
export function scheduleTable(m: number, vpp: number, groupSize: number): { mb: number; chunk: number }[] {
  const table: { mb: number; chunk: number }[] = [];
  for (let lo = 0; lo < m; lo += groupSize) {
    const hi = Math.min(lo + groupSize, m);
    for (let chunk = 0; chunk < vpp; chunk++) {
      for (let mb = lo; mb < hi; mb++) table.push({ mb, chunk });
    }
  }
  return table;
}

/** Copied from Megatron-LM `get_pp_rank_microbatches` (number of warmup virtual micro-batches). */
export function numWarmup(m: number, pp: number, rank: number, vpp: number, groupSize: number): number {
  const total = m * vpp;
  if (pp === 1) return Math.min(1, total);
  const w = (pp - rank - 1) * 2 + (vpp - 1) * groupSize;
  return Math.min(w, total);
}

/**
 * Megatron-LM interleaved 1F1B (`forward_backward_pipelining_with_interleaving`,
 * synchronous p2p path, i.e. `overlap_p2p_comm = False`).
 *
 * Communication is issued exactly where Megatron issues it:
 *   warmup   : F(k); send_forward_recv_forward (+ recv_backward on the last warmup step)
 *   steady   : F(fk); B(bk); send_forward_backward_recv_forward_backward
 *   cooldown : B(k); send_backward_recv_backward
 * Note the forward output of a steady-state iteration is only sent *after*
 * the backward pass; this is a real property of the synchronous path.
 *
 * The decision "does the next F / B on this rank need an input from a peer"
 * is computed from the topology, which for the default group size (pp)
 * coincides with Megatron's leading-stage alignment logic.
 */
export function interleavedProgram(cfg: SimConfig): Program {
  const { pp, vpp, microBatches: m } = cfg;
  const groupSize = pp;
  if (m % pp !== 0) {
    throw new Error(`interleaved-1f1b requires microBatches (${m}) to be divisible by pp (${pp})`);
  }
  const topo = new Topology(pp, vpp);
  const table = scheduleTable(m, vpp, groupSize);
  const total = m * vpp;
  const program: Program = [];

  for (let rank = 0; rank < pp; rank++) {
    const steps: Step[] = [];
    const F = (v: number) => table[v];
    const B = (v: number) => ({ mb: table[v].mb, chunk: vpp - 1 - table[v].chunk });
    const recvF = (v: number) => (v < total ? topo.recvF(rank, F(v).mb, F(v).chunk) : null);
    const recvB = (v: number) => (v < total ? topo.recvB(rank, B(v).mb, B(v).chunk) : null);
    const sendF = (v: number) => topo.sendF(rank, F(v).mb, F(v).chunk);
    const sendB = (v: number) => topo.sendB(rank, B(v).mb, B(v).chunk);

    const warmup = numWarmup(m, pp, rank, vpp, groupSize);
    const allWarmup = warmup === total;
    const remaining = total - warmup;

    // input_tensors[0].append(recv_forward(...))
    push(steps, comm([], [recvF(0)]));

    for (let k = 0; k < warmup; k++) {
      steps.push(compute('F', F(k).mb, F(k).chunk));
      const recvs = [recvF(k + 1)];
      if (k === warmup - 1 && !allWarmup) recvs.push(recvB(0));
      push(steps, comm([sendF(k)], recvs));
    }

    if (allWarmup) {
      // output_tensor_grads[num_model_chunks - 1].append(recv_backward(...))
      push(steps, comm([], [recvB(0)]));
    }

    for (let k = 0; k < remaining; k++) {
      const fk = k + warmup;
      const bk = k;
      steps.push(compute('F', F(fk).mb, F(fk).chunk));
      steps.push(compute('B', B(bk).mb, B(bk).chunk));
      const nextF = k === remaining - 1 ? null : recvF(fk + 1);
      push(steps, comm([sendF(fk), sendB(bk)], [nextF, recvB(bk + 1)]));
    }

    for (let k = remaining; k < total; k++) {
      steps.push(compute('B', B(k).mb, B(k).chunk));
      push(steps, comm([sendB(k)], [recvB(k + 1)]));
    }

    program.push(steps);
  }
  return program;
}

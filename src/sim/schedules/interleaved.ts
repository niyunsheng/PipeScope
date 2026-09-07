import type { Program, SimConfig, Step } from '../types.ts';
import { Topology, comm, compute, post, push, wait } from './common.ts';

/**
 * Schedule lookup table, copied from Megatron-LM `get_schedule_table` (core_v0.19.0, L962).
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

/** Copied from Megatron-LM `get_pp_rank_microbatches` (core_v0.19.0, L902): number of warmup virtual micro-batches. */
export function numWarmup(m: number, pp: number, rank: number, vpp: number, groupSize: number): number {
  const total = m * vpp;
  if (pp === 1) return Math.min(1, total);
  const w = (pp - rank - 1) * 2 + (vpp - 1) * groupSize;
  return Math.min(w, total);
}

/**
 * Megatron-LM interleaved 1F1B (`forward_backward_pipelining_with_interleaving`).
 * `commModel = 'sync'` selects the synchronous p2p path (`overlap_p2p_comm =
 * False`, blocking batched calls); `'async'` selects the overlap path (isend /
 * irecv, waits right before use). See `InterleavedKnobs`.
 * Reference: Megatron-LM tag core_v0.19.0, megatron/core/pipeline_parallel/schedules.py
 *   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.0/megatron/core/pipeline_parallel/schedules.py#L992
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
  const G = cfg.groupSize; // Megatron: microbatch_group_size_per_vp_stage, default pp
  return interleavedFamily(cfg, { groupSize: G, warmupOf: (r) => numWarmup(m, pp, r, vpp, G), sendAfter: cfg.sendAfter, waitGrad: cfg.waitGrad, nonblocking: cfg.commModel === 'async' });
}

/** Knobs of the interleaved skeleton that the custom schedule exposes. */
export interface InterleavedKnobs {
  /** Micro-batches processed on one chunk before switching (Megatron: pp). */
  groupSize: number;
  /** Warmup forwards for a rank, before clamping to the total. */
  warmupOf: (rank: number) => number;
  /** Steady state: send the forward's output right after it (`F`) or after the backward (`B`). */
  sendAfter: 'F' | 'B';
  /**
   * Steady state: wait for a backward's gradient before the preceding forward
   * (`beforeF`, in the batched step that ends the previous round) or right
   * before the backward (`beforeB`). The next forward's input is always waited
   * for right before that forward.
   */
  waitGrad: 'beforeF' | 'beforeB';
  /**
   * `false`: every communication call posts and waits (`comm` steps), i.e.
   * Megatron's synchronous path with `sendAfter = 'B'`, `waitGrad = 'beforeF'`.
   * `true`: isend / irecv. Warmup and cooldown keep blocking pairs (as in
   * Megatron's `overlap_p2p_comm` path without warmup-flush overlap); in the
   * steady state sends are posted right after the op that produced them,
   * receives for the next round are posted together with them, and the rank
   * waits only right before the consuming op. With `sendAfter = 'F'`,
   * `waitGrad = 'beforeB'` this is Megatron's overlap path.
   */
  nonblocking: boolean;
}

/**
 * Interleaved 1F1B skeleton with the knobs above. Megatron's synchronous
 * schedule is `groupSize = pp`, `warmupOf = numWarmup`, `sendAfter = 'B'`,
 * `waitGrad = 'beforeF'`.
 */
export function interleavedFamily(cfg: SimConfig, knobs: InterleavedKnobs): Program {
  const { pp, vpp, microBatches: m } = cfg;
  const { groupSize, sendAfter, waitGrad, nonblocking } = knobs;
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

    const warmup = Math.max(0, Math.min(total, Math.round(knobs.warmupOf(rank))));
    const allWarmup = warmup === total;
    const remaining = total - warmup;

    // input_tensors[0].append(recv_forward(...))
    push(steps, comm([], [recvF(0)]));

    if (nonblocking) {
      // isend / irecv: post early, wait right before use. Warmup and cooldown
      // stay blocking pairs like Megatron's overlap path (path B).
      for (let k = 0; k < warmup; k++) {
        steps.push(compute('F', F(k).mb, F(k).chunk));
        push(steps, comm([sendF(k)], [recvF(k + 1)]));
        // Last warmup step: irecv for the first backward's gradient, waited for later.
        if (k === warmup - 1 && !allWarmup) push(steps, post([], [recvB(0)]));
      }
      if (allWarmup) push(steps, comm([], [recvB(0)]));
      for (let k = 0; k < remaining; k++) {
        const fk = k + warmup;
        const bk = k;
        const nextF = k === remaining - 1 ? null : recvF(fk + 1);
        if (k > 0) push(steps, wait([recvF(fk)])); // input posted in the previous round
        if (waitGrad === 'beforeF') push(steps, wait([recvB(bk)]));
        steps.push(compute('F', F(fk).mb, F(fk).chunk));
        if (sendAfter === 'F') push(steps, post([sendF(fk)], [nextF]));
        if (waitGrad === 'beforeB') push(steps, wait([recvB(bk)]));
        steps.push(compute('B', B(bk).mb, B(bk).chunk));
        push(steps, post([sendB(bk), sendAfter === 'B' ? sendF(fk) : null], [recvB(bk + 1), sendAfter === 'B' ? nextF : null]));
      }
      for (let k = remaining; k < total; k++) {
        push(steps, wait([recvB(k)])); // no-op unless it was posted (first cooldown step)
        steps.push(compute('B', B(k).mb, B(k).chunk));
        push(steps, comm([sendB(k)], [recvB(k + 1)]));
      }
      program.push(steps);
      continue;
    }

    if (waitGrad === 'beforeB') {
      // Blocking pairs with the gradient waited for right before the backward
      // that needs it (1F1B's pairing generalised to chunks). With
      // `sendAfter = 'F'` this is `forward_backward_pipelining_without_
      // interleaving` when vpp = 1.
      for (let k = 0; k < warmup; k++) {
        steps.push(compute('F', F(k).mb, F(k).chunk));
        push(steps, comm([sendF(k)], [recvF(k + 1)]));
      }
      if (allWarmup) push(steps, comm([], [recvB(0)]));
      for (let k = 0; k < remaining; k++) {
        const fk = k + warmup;
        const bk = k;
        steps.push(compute('F', F(fk).mb, F(fk).chunk));
        push(steps, comm([sendAfter === 'F' ? sendF(fk) : null], [recvB(bk)]));
        steps.push(compute('B', B(bk).mb, B(bk).chunk));
        const last = k === remaining - 1;
        // The last steady backward also fetches the first cooldown gradient.
        push(steps, comm([sendAfter === 'B' ? sendF(fk) : null, sendB(bk)], [last ? recvB(bk + 1) : recvF(fk + 1)]));
      }
      for (let k = remaining; k < total; k++) {
        steps.push(compute('B', B(k).mb, B(k).chunk));
        push(steps, comm([sendB(k)], [recvB(k + 1)]));
      }
      program.push(steps);
      continue;
    }

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
      if (sendAfter === 'F') push(steps, comm([sendF(fk)], []));
      steps.push(compute('B', B(bk).mb, B(bk).chunk));
      const nextF = k === remaining - 1 ? null : recvF(fk + 1);
      push(steps, comm([sendAfter === 'B' ? sendF(fk) : null, sendB(bk)], [nextF, recvB(bk + 1)]));
    }

    for (let k = remaining; k < total; k++) {
      steps.push(compute('B', B(k).mb, B(k).chunk));
      push(steps, comm([sendB(k)], [recvB(k + 1)]));
    }

    program.push(steps);
  }
  return program;
}

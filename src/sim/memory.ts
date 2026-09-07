import type { CostModel } from './cost.ts';
import type { MemorySample, Op, TransferRecord } from './types.ts';
import { fTag } from './schedules/common.ts';

/**
 * Activation memory model.
 *
 * For every (mb, chunk) on a rank:
 *   - the *input* tensor is allocated when the recv is posted: Megatron's
 *     `_communicate` allocates the destination tensor before the irecv, so
 *     the buffer exists from then on. For stage 0 the input appears at the
 *     forward start;
 *   - the *intermediate* activations are allocated when the forward starts;
 *   - both are released when the backward pass ends;
 *   - on the *sending* side, an op's output tensor (activation for F, input
 *     gradient for B) stays alive from the op's end until the transfer has
 *     landed, when Megatron's `deallocate_pipeline_outputs` frees it.
 * The static baseline (weights, gradients, optimizer state) is a constant.
 * The result per rank is a step function sampled at every event.
 */
export function computeMemory(
  ops: Op[],
  pp: number,
  cost: CostModel,
  baseline: number,
  transfers: TransferRecord[] = [],
): MemorySample[][] {
  const allocated = new Map(transfers.map((tr) => [tr.tag, tr.recvPosted]));
  interface Ev {
    t: number;
    delta: number;
    key: string;
    event: MemorySample['event'];
    /** Releases sort before allocations at equal timestamps. */
    order: number;
  }
  const events: Ev[][] = Array.from({ length: pp }, () => []);
  for (const op of ops) {
    const key = `${op.mb}:${op.chunk}`;
    if (op.kind === 'F') {
      const input = cost.activationInput(op.mb, op.chunk, op.rank);
      const inter = cost.activationIntermediate(op.mb, op.chunk, op.rank);
      const allocAt = op.stage > 0 ? allocated.get(fTag(op.mb, op.stage - 1)) : undefined;
      const tIn = Math.min(op.start, allocAt ?? op.start);
      if (input > 0) events[op.rank].push({ t: tIn, delta: input, key, event: 'input', order: 1 });
      if (inter > 0) events[op.rank].push({ t: op.start, delta: inter, key, event: 'forward', order: 2 });
    } else if (op.kind === 'B') {
      const total = cost.activationInput(op.mb, op.chunk, op.rank) + cost.activationIntermediate(op.mb, op.chunk, op.rank);
      events[op.rank].push({ t: op.end, delta: -total, key, event: 'release', order: 0 });
    }
  }
  // Sender-side output buffers: held from the producing op's end until the data has landed.
  for (const tr of transfers) {
    if (tr.producer === null) continue;
    const prod = ops[tr.producer];
    const bytes = cost.activationInput(tr.mb, prod.chunk, tr.from);
    if (bytes <= 0 || tr.landed <= prod.end) continue;
    const key = `${tr.mb}:${prod.chunk}:${tr.kind === 'F' ? 'out' : 'grad'}`;
    events[tr.from].push({ t: prod.end, delta: bytes, key, event: 'output', order: 2 });
    events[tr.from].push({ t: tr.landed, delta: -bytes, key, event: 'sent', order: 0 });
  }
  return events.map((evs) => {
    evs.sort((a, b) => a.t - b.t || a.order - b.order);
    const samples: MemorySample[] = [{ t: 0, bytes: baseline, event: 'release', resident: [] }];
    let bytes = baseline;
    const resident = new Set<string>();
    for (const ev of evs) {
      bytes += ev.delta;
      if (ev.delta > 0) resident.add(ev.key);
      else resident.delete(ev.key);
      samples.push({ t: ev.t, bytes, event: ev.event, resident: [...resident] });
    }
    return samples;
  });
}

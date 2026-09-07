import type { CostModel } from './cost.ts';
import type { MemorySample, Op, TransferRecord } from './types.ts';
import { fTag } from './schedules/common.ts';

/**
 * Activation memory model.
 *
 * For every (mb, chunk) on a rank:
 *   - the *input* tensor is allocated when its receive buffer comes into
 *     existence: the earlier of the recv being posted (Megatron's
 *     `_communicate` allocates the destination tensor before the irecv) and
 *     the data starting to move towards this rank (under the async model the
 *     sender writes into the receiver's buffer before the recv is posted).
 *     This is exactly where the recv bar begins in the timeline, and always
 *     at or before the forward. For stage 0 the input appears at the forward start;
 *   - the *intermediate* activations are allocated when the forward starts;
 *   - both are released when the backward pass ends.
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
  const allocated = new Map(transfers.map((tr) => [tr.tag, Math.min(tr.recvPosted, tr.start)]));
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

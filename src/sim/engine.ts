import type { CostModel } from './cost.ts';
import type {
  BlockedBy,
  CommModel,
  CommStep,
  IdleInterval,
  Op,
  Program,
  SimFailure,
  Transfer,
  TransferRecord,
} from './types.ts';

export interface RunResult {
  ops: Op[];
  idles: IdleInterval[];
  /** Time at which each rank finished its program. */
  rankFinish: number[];
  /**
   * Every transfer with its posting, start and landing times. Under `async`
   * data can land before the recv completes (it is buffered); under `sync`
   * landing equals the recv completion.
   */
  transfers: TransferRecord[];
  /** Set when the run stopped early; the other fields hold the partial result. */
  failure: SimFailure | null;
}

/** One posted end of a transfer, waiting for its counterpart. */
interface Posted {
  rank: number;
  time: number;
  transfer: Transfer;
}

/**
 * FIFO channel for one (src, dst, kind) triple. The i-th send matches the
 * i-th recv, exactly like untagged NCCL point-to-point traffic. Tags are only
 * used to assert that the schedule generator paired them up correctly.
 */
interface Channel {
  sends: Posted[];
  recvs: Posted[];
}

interface RankState {
  pc: number;
  clock: number;
  /** Set while blocked in a comm step: index into channel queues per transfer. */
  pending: { channel: Channel; index: number; transfer: Transfer; isSend: boolean }[] | null;
  /** Ids of compute ops issued by this rank, in order. */
  opIds: number[];
  /** Binding cross-rank predecessors found by the last comm step; consumed by the next compute op. */
  commBinding: BlockedBy[];
}

function channelKey(src: number, dst: number, kind: string): string {
  return `${src}->${dst}:${kind}`;
}

/**
 * Execute per-rank programs under rendezvous communication semantics.
 *
 * Each rank advances through its steps; compute steps simply consume time,
 * comm steps block the rank until every transfer in the step has completed.
 * Under `sync` a transfer completes at `max(sender arrival, receiver arrival)
 * + transfer time` for both peers; under `async` the sender completes at once
 * and the receiver completes at `max(receiver arrival, sender arrival +
 * transfer time)`. See `CommModel` in types.ts.
 * Because all programs are static, a fixed-point sweep over runnable ranks is
 * equivalent to an event-heap DES and easier to reason about. A sweep that
 * makes no progress while some rank is still blocked means the schedule
 * deadlocks (e.g. an unmatched recv), which is reported with context.
 */
export function runProgram(program: Program, pp: number, cost: CostModel, commModel: CommModel = 'async'): RunResult {
  const ranks: RankState[] = Array.from({ length: pp }, () => ({ pc: 0, clock: 0, pending: null, opIds: [], commBinding: [] }));
  const channels = new Map<string, Channel>();
  const ops: Op[] = [];
  const idles: IdleInterval[] = [];
  /** (mb, stage) pairs whose forward has been issued, for the local F -> B check below. */
  const forwarded = new Set<string>();
  const transfers: TransferRecord[] = [];
  let nextOpId = 0;
  let failure: SimFailure | null = null;
  /** Describe what every blocked rank is waiting on, for failure reports. */
  const blockedRanks = () =>
    ranks.flatMap((st, r) =>
      st.pending
        ? [{ rank: r, since: st.clock, waiting: st.pending.map((p) => `${p.isSend ? 'send' : 'recv'} ${p.transfer.tag} ${p.isSend ? 'to' : 'from'} rank ${p.transfer.peer}`).join(', ') }]
        : [],
    );

  const getChannel = (src: number, dst: number, kind: string): Channel => {
    const key = channelKey(src, dst, kind);
    let ch = channels.get(key);
    if (!ch) {
      ch = { sends: [], recvs: [] };
      channels.set(key, ch);
    }
    return ch;
  };

  /** Id of the latest compute op on `rank` that ended at or before `time`, or null. */
  const lastOpBefore = (rank: number, time: number): number | null => {
    const ids = ranks[rank].opIds;
    for (let i = ids.length - 1; i >= 0; i--) {
      if (ops[ids[i]].end <= time + 1e-9) return ids[i];
    }
    return null;
  };

  /** The compute op on `rank` that produced `t` (same kind and mb, latest one finished by `time`). */
  const producerOf = (rank: number, t: Transfer, time: number): number | null => {
    const ids = ranks[rank].opIds;
    for (let i = ids.length - 1; i >= 0; i--) {
      const o = ops[ids[i]];
      if (o.kind === t.kind && o.mb === t.mb && o.end <= time + 1e-9) return o.id;
    }
    return null;
  };

  const post = (rank: number, step: CommStep): void => {
    const st = ranks[rank];
    st.pending = [];
    for (const t of step.sends) {
      const ch = getChannel(rank, t.peer, t.kind);
      ch.sends.push({ rank, time: st.clock, transfer: t });
      st.pending.push({ channel: ch, index: ch.sends.length - 1, transfer: t, isSend: true });
    }
    for (const t of step.recvs) {
      const ch = getChannel(t.peer, rank, t.kind);
      ch.recvs.push({ rank, time: st.clock, transfer: t });
      st.pending.push({ channel: ch, index: ch.recvs.length - 1, transfer: t, isSend: false });
    }
  };

  /** Try to complete the comm step a rank is blocked on. Returns true on progress. */
  const tryComplete = (rank: number): boolean => {
    const st = ranks[rank];
    if (!st.pending) return false;
    let completion = st.clock;
    let peerArrival = st.clock;
    let last: { transfer: Transfer; isSend: boolean; peer: Posted } | null = null;
    const results: { done: number; isSend: boolean; peer: Posted; wire: number; transfer: Transfer; from: number; to: number }[] = [];
    for (const p of st.pending) {
      if (commModel === 'async' && p.isSend) continue; // buffered send: never blocks the sender
      const counterpart = p.isSend ? p.channel.recvs[p.index] : p.channel.sends[p.index];
      if (!counterpart) return false; // peer has not reached the matching step yet
      if (counterpart.transfer.tag !== p.transfer.tag) {
        failure = {
          kind: 'tag-mismatch',
          message:
            `Tag mismatch on rank ${rank}: ${p.isSend ? 'send' : 'recv'} ${p.transfer.tag} ` +
            `paired with peer ${counterpart.rank}'s ${counterpart.transfer.tag}. ` +
            'The schedule generator posted transfers in an inconsistent order.',
          blocked: blockedRanks(),
          op: null,
        };
        return false;
      }
      const from = p.isSend ? rank : p.transfer.peer;
      const to = p.isSend ? p.transfer.peer : rank;
      const wire = cost.transfer(p.transfer.kind, p.transfer.mb, from, to);
      // sync: data moves only after both peers posted. async: data was already
      // in flight since the sender posted, the receiver just waits for it to land.
      const done = commModel === 'sync' ? Math.max(st.clock, counterpart.time) + wire : Math.max(st.clock, counterpart.time + wire);
      results.push({ done, isSend: p.isSend, peer: counterpart, wire, transfer: p.transfer, from, to });
      if (done >= completion) {
        completion = done;
        // Time until the peer posted its end (0 if it was already there).
        // Under async the send was posted at counterpart.time as well, so the
        // same expression tells how long this rank waited for the peer to
        // even start sending; the remainder of the idle is wire time.
        peerArrival = Math.max(st.clock, counterpart.time);
        last = { transfer: p.transfer, isSend: p.isSend, peer: counterpart };
      }
    }
    // All counterparts are present: the step completes now. Only from here on
    // may state be recorded, since an early `return false` above means the
    // whole loop runs again on a later sweep.
    for (const r of results) {
      if (r.isSend) continue;
      // Recorded once, from the receiving side, which sees both posting times.
      const start = commModel === 'sync' ? Math.max(st.clock, r.peer.time) : r.peer.time;
      transfers.push({
        tag: r.transfer.tag,
        kind: r.transfer.kind,
        mb: r.transfer.mb,
        from: r.from,
        to: r.to,
        sendPosted: r.peer.time,
        recvPosted: st.clock,
        start,
        landed: start + r.wire,
        producer: producerOf(r.from, r.transfer, r.peer.time),
      });
    }
    if (completion > st.clock && last) {
      const t = last.transfer;
      idles.push({
        rank,
        start: st.clock,
        end: completion,
        reason: last.isSend ? 'wait-send' : 'wait-recv',
        transferTag: t.tag,
        peerWait: peerArrival - st.clock,
        transfer: completion - peerArrival,
      });
    }
    // Every transfer that completed exactly at the step's completion is a
    // binding constraint on the next op, whether or not this rank idled. The
    // one chosen as `last` goes first so it stays the chain's primary hop.
    st.commBinding = [];
    const primary = last ? results.find((r) => r.peer === last!.peer) : undefined;
    const ordered = primary ? [primary, ...results.filter((r) => r !== primary)] : results;
    const arrival = st.clock;
    const ownPrev = st.opIds.length ? st.opIds[st.opIds.length - 1] : null;
    for (const r of ordered) {
      if (Math.abs(r.done - completion) > 1e-9) continue;
      // Who set the completion time: the peer (it posted after we arrived, or
      // its data was still in flight), or ourselves (we arrived last and only
      // the wire time remained)? In the latter case the constraint is our own
      // previous op plus wire, so it is attributed to that op.
      const peerBound = commModel === 'sync' ? r.peer.time >= arrival - 1e-9 : r.peer.time + r.wire >= arrival - 1e-9;
      const producer = peerBound ? lastOpBefore(r.peer.rank, r.peer.time) : ownPrev;
      if (producer !== null && !st.commBinding.some((b) => b.op === producer)) {
        st.commBinding.push({ reason: r.isSend ? 'wait-send' : 'wait-recv', op: producer, transferTag: r.transfer.tag });
      }
    }
    st.clock = completion;
    st.pending = null;
    st.pc += 1;
    return true;
  };

  for (;;) {
    let progress = false;
    let unfinished = false;
    for (let r = 0; r < pp; r++) {
      const st = ranks[r];
      if (st.pending) {
        unfinished = true;
        progress = tryComplete(r) || progress;
        continue;
      }
      if (st.pc >= program[r].length) continue;
      unfinished = true;
      const step = program[r][st.pc];
      if (step.type === 'compute') {
        // Cross-rank dependencies are enforced by the rendezvous below (a tensor
        // cannot be received before it was sent). The one dependency no
        // communication covers is a backward needing its own rank's forward,
        // which matters at the last stage where the gradient is produced
        // locally. Megatron fails here with an empty `input_tensors` list; we
        // fail with a message instead of drawing a plausible-looking timeline.
        const key = `${step.mb}:${step.chunk * pp + r}`;
        const dur = cost.compute(step.kind, step.mb, step.chunk, r);
        if (step.kind === 'F') forwarded.add(key);
        else if (!forwarded.has(key)) {
          failure = {
            kind: 'program',
            message: `Program error on rank ${r}: B of micro-batch ${step.mb} at stage ${step.chunk * pp + r} is scheduled before its forward.`,
            blocked: blockedRanks(),
            op: { id: nextOpId, rank: r, chunk: step.chunk, stage: step.chunk * pp + r, mb: step.mb, kind: 'B', start: st.clock, end: st.clock + dur, predecessors: [] },
          };
          break;
        }
        const prev = st.opIds.length ? st.opIds[st.opIds.length - 1] : null;
        const predecessors: BlockedBy[] = st.commBinding;
        st.commBinding = [];
        // Program order binds only when the previous op ended exactly now.
        if (prev !== null && Math.abs(ops[prev].end - st.clock) < 1e-9 && !predecessors.some((b) => b.op === prev)) {
          predecessors.push({ reason: 'program', op: prev });
        }
        const id = nextOpId++;
        ops.push({
          id,
          rank: r,
          chunk: step.chunk,
          stage: step.chunk * pp + r,
          mb: step.mb,
          kind: step.kind,
          start: st.clock,
          end: st.clock + dur,
          predecessors,
        });
        st.opIds.push(id);
        st.clock += dur;
        st.pc += 1;
      } else {
        if (step.sends.length === 0 && step.recvs.length === 0) {
          st.pc += 1;
        } else {
          post(r, step);
          // Complete immediately if the peers are already there.
          tryComplete(r);
        }
      }
      progress = true;
    }
    if (failure) break;
    if (!unfinished) break;
    if (!progress) {
      const blocked = blockedRanks();
      failure = {
        kind: 'deadlock',
        message: `Deadlock: no rank can make progress. ${blocked.map((b) => `rank ${b.rank} @t=${b.since}: ${b.waiting}`).join('; ')}`,
        blocked,
        op: null,
      };
      break;
    }
  }

  return { ops, idles, rankFinish: ranks.map((st) => st.clock), transfers, failure };
}

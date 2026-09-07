import type { CostModel } from './cost.ts';
import type {
  BlockedBy,
  CommStep,
  IdleInterval,
  Op,
  PostStep,
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
  /** Every transfer with its posting, rendezvous and landing times. */
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

/** One end of a transfer this rank has posted and not yet waited for. */
interface Outstanding {
  channel: Channel;
  index: number;
  transfer: Transfer;
  isSend: boolean;
  postedAt: number;
}

interface RankState {
  pc: number;
  clock: number;
  /** Transfers posted by this rank whose completion has not been waited for yet, by tag. */
  outstanding: Map<string, Outstanding>;
  /** Every tag this rank has ever posted, to tell "already completed" from "never posted" in a wait. */
  everPosted: Set<string>;
  /** Set while blocked in a comm / wait step: the tags being waited for. */
  waiting: string[] | null;
  /** Ids of compute ops issued by this rank, in order. */
  opIds: number[];
  /** Binding cross-rank predecessors found by the last blocking step; consumed by the next compute op. */
  commBinding: BlockedBy[];
}

function channelKey(src: number, dst: number, kind: string): string {
  return `${src}->${dst}:${kind}`;
}

/**
 * Execute per-rank programs under NCCL-style rendezvous communication.
 *
 * Each rank advances through its steps. Compute steps consume time. A
 * transfer starts moving once both peers have posted their end and lands
 * `latency` later, for both peers alike. `comm` steps post and block until
 * everything in them has landed (Megatron's synchronous path); `post` steps
 * only post (isend / irecv) and `wait` steps block until previously posted
 * transfers have landed (the overlap path). Because all programs are static,
 * a fixed-point sweep over runnable ranks is equivalent to an event-heap DES
 * and easier to reason about. A sweep that makes no progress while some rank
 * is still blocked means the schedule deadlocks, which is reported with the
 * partial timeline.
 */
export function runProgram(program: Program, pp: number, cost: CostModel): RunResult {
  const ranks: RankState[] = Array.from({ length: pp }, () => ({
    pc: 0,
    clock: 0,
    outstanding: new Map(),
    everPosted: new Set(),
    waiting: null,
    opIds: [],
    commBinding: [],
  }));
  const channels = new Map<string, Channel>();
  const ops: Op[] = [];
  const idles: IdleInterval[] = [];
  /** (mb, stage) pairs whose forward has been issued, for the local F -> B check below. */
  const forwarded = new Set<string>();
  const transfers: TransferRecord[] = [];
  const recorded = new Set<string>();
  let nextOpId = 0;
  let failure: SimFailure | null = null;

  const describe = (o: Outstanding) => `${o.isSend ? 'send' : 'recv'} ${o.transfer.tag} ${o.isSend ? 'to' : 'from'} rank ${o.transfer.peer}`;
  /** Describe what every blocked rank is waiting on, for failure reports. */
  const blockedRanks = () =>
    ranks.flatMap((st, r) =>
      st.waiting
        ? [{ rank: r, since: st.clock, waiting: st.waiting.map((tag) => describe(st.outstanding.get(tag)!)).join(', ') }]
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

  /** Post every transfer of a comm / post step at the rank's current clock. */
  const postAll = (rank: number, step: CommStep | PostStep): string[] => {
    const st = ranks[rank];
    const tags: string[] = [];
    for (const t of step.sends) {
      const ch = getChannel(rank, t.peer, t.kind);
      ch.sends.push({ rank, time: st.clock, transfer: t });
      st.outstanding.set(t.tag, { channel: ch, index: ch.sends.length - 1, transfer: t, isSend: true, postedAt: st.clock });
      st.everPosted.add(t.tag);
      tags.push(t.tag);
    }
    for (const t of step.recvs) {
      const ch = getChannel(t.peer, rank, t.kind);
      ch.recvs.push({ rank, time: st.clock, transfer: t });
      st.outstanding.set(t.tag, { channel: ch, index: ch.recvs.length - 1, transfer: t, isSend: false, postedAt: st.clock });
      st.everPosted.add(t.tag);
      tags.push(t.tag);
    }
    return tags;
  };

  /**
   * Try to finish the blocking step a rank is in. Every waited-for transfer
   * needs its counterpart posted; then it lands at `max(both posted) + wire`.
   * Returns true when the rank could move on.
   */
  const tryComplete = (rank: number): boolean => {
    const st = ranks[rank];
    if (!st.waiting) return false;
    const results: { done: number; o: Outstanding; peer: Posted }[] = [];
    for (const tag of st.waiting) {
      const o = st.outstanding.get(tag);
      if (!o) {
        failure = { kind: 'program', message: `Program error on rank ${rank}: waiting for ${tag}, which was never posted.`, blocked: blockedRanks(), op: null };
        return false;
      }
      const counterpart = o.isSend ? o.channel.recvs[o.index] : o.channel.sends[o.index];
      if (!counterpart) return false; // peer has not posted the matching end yet
      if (counterpart.transfer.tag !== o.transfer.tag) {
        failure = {
          kind: 'tag-mismatch',
          message:
            `Tag mismatch on rank ${rank}: ${describe(o)} paired with peer ${counterpart.rank}'s ${counterpart.transfer.tag}. ` +
            'The schedule generator posted transfers in an inconsistent order.',
          blocked: blockedRanks(),
          op: null,
        };
        return false;
      }
      const from = o.isSend ? rank : o.transfer.peer;
      const to = o.isSend ? o.transfer.peer : rank;
      const wire = cost.transfer(o.transfer.kind, o.transfer.mb, from, to);
      const start = Math.max(o.postedAt, counterpart.time);
      results.push({ done: start + wire, o, peer: counterpart });
      if (!recorded.has(o.transfer.tag)) {
        recorded.add(o.transfer.tag);
        const sendPosted = o.isSend ? o.postedAt : counterpart.time;
        const recvPosted = o.isSend ? counterpart.time : o.postedAt;
        transfers.push({
          tag: o.transfer.tag,
          kind: o.transfer.kind,
          mb: o.transfer.mb,
          from,
          to,
          sendPosted,
          recvPosted,
          start,
          landed: start + wire,
          producer: producerOf(from, o.transfer, sendPosted),
        });
      }
    }
    // All counterparts present: the step completes now.
    let completion = st.clock;
    let last: { done: number; o: Outstanding; peer: Posted } | null = null;
    for (const r of results) {
      if (r.done >= completion) {
        completion = r.done;
        last = r;
      }
    }
    if (completion > st.clock && last) {
      const peerArrival = Math.max(st.clock, last.peer.time);
      idles.push({
        rank,
        start: st.clock,
        end: completion,
        reason: last.o.isSend ? 'wait-send' : 'wait-recv',
        transferTag: last.o.transfer.tag,
        peerWait: peerArrival - st.clock,
        transfer: completion - peerArrival,
      });
    }
    // Every transfer that completed exactly at the step's completion is a
    // binding constraint on the next op, whether or not this rank idled. The
    // one chosen as `last` goes first so it stays the chain's primary hop.
    st.commBinding = [];
    const ordered = last ? [last, ...results.filter((r) => r !== last)] : results;
    const ownPrev = st.opIds.length ? st.opIds[st.opIds.length - 1] : null;
    for (const r of ordered) {
      if (Math.abs(r.done - completion) > 1e-9) continue;
      // Who set the completion: the peer (posted after we arrived, so the
      // rendezvous waited for it) or ourselves (we arrived last and only the
      // wire time remained; attributed to our own previous op).
      const peerBound = r.peer.time >= st.clock - 1e-9;
      const producer = peerBound ? lastOpBefore(r.peer.rank, r.peer.time) : ownPrev;
      if (producer !== null && !st.commBinding.some((b) => b.op === producer)) {
        st.commBinding.push({ reason: r.o.isSend ? 'wait-send' : 'wait-recv', op: producer, transferTag: r.o.transfer.tag });
      }
    }
    for (const tag of st.waiting) st.outstanding.delete(tag);
    st.clock = completion;
    st.waiting = null;
    st.pc += 1;
    return true;
  };

  for (;;) {
    let progress = false;
    let unfinished = false;
    for (let r = 0; r < pp; r++) {
      const st = ranks[r];
      if (st.waiting) {
        unfinished = true;
        progress = tryComplete(r) || progress;
        if (failure) break;
        continue;
      }
      if (st.pc >= program[r].length) continue;
      unfinished = true;
      const step = program[r][st.pc];
      if (step.type === 'compute') {
        // Cross-rank dependencies are enforced by the rendezvous (a tensor
        // cannot be received before it was sent). The one dependency no
        // communication covers is a backward needing its own rank's forward,
        // which matters at the last stage where the gradient is produced
        // locally. Megatron fails here with an empty `input_tensors` list; we
        // fail with a message instead of drawing a plausible-looking timeline.
        const key = `${step.mb}:${step.chunk * pp + r}`;
        const dur = cost.compute(step.kind, step.mb, step.chunk, r);
        if (step.kind === 'F') forwarded.add(key);
        else if (step.kind === 'B' && !forwarded.has(key)) {
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
      } else if (step.type === 'post') {
        postAll(r, step);
        st.pc += 1;
      } else {
        // `comm` posts then waits; `wait` waits for tags posted earlier. Tags
        // already waited for (hence no longer outstanding) are no-ops; a tag
        // this rank never posted is a schedule bug.
        if (step.type === 'wait') {
          const unknown = step.tags.find((tag) => !st.everPosted.has(tag));
          if (unknown !== undefined) {
            failure = { kind: 'program', message: `Program error on rank ${r}: waiting for ${unknown}, which was never posted.`, blocked: blockedRanks(), op: null };
            break;
          }
        }
        const tags = step.type === 'comm' ? postAll(r, step) : step.tags.filter((tag) => st.outstanding.has(tag));
        if (tags.length === 0) {
          st.pc += 1;
        } else {
          st.waiting = tags;
          tryComplete(r); // completes at once if the peers are already there
          if (failure) break;
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

import type { CostModel } from './cost.ts';
import type {
  CommModel,
  CommStep,
  IdleInterval,
  Op,
  Program,
  Transfer,
} from './types.ts';

export interface RunResult {
  ops: Op[];
  idles: IdleInterval[];
  /** Time at which each rank finished its program. */
  rankFinish: number[];
  /**
   * When each received tensor landed on the receiving rank, keyed by tag.
   * Under `async` this can be earlier than the recv completes (data is
   * buffered); under `sync` it equals the recv completion.
   */
  landed: Map<string, number>;
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
  const ranks: RankState[] = Array.from({ length: pp }, () => ({ pc: 0, clock: 0, pending: null }));
  const channels = new Map<string, Channel>();
  const ops: Op[] = [];
  const idles: IdleInterval[] = [];
  const landed = new Map<string, number>();
  let nextOpId = 0;

  const getChannel = (src: number, dst: number, kind: string): Channel => {
    const key = channelKey(src, dst, kind);
    let ch = channels.get(key);
    if (!ch) {
      ch = { sends: [], recvs: [] };
      channels.set(key, ch);
    }
    return ch;
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
    let last: { transfer: Transfer; isSend: boolean; peerTime: number; wire: number } | null = null;
    for (const p of st.pending) {
      if (commModel === 'async' && p.isSend) continue; // buffered send: never blocks the sender
      const counterpart = p.isSend ? p.channel.recvs[p.index] : p.channel.sends[p.index];
      if (!counterpart) return false; // peer has not reached the matching step yet
      if (counterpart.transfer.tag !== p.transfer.tag) {
        throw new Error(
          `Tag mismatch on rank ${rank}: ${p.isSend ? 'send' : 'recv'} ${p.transfer.tag} ` +
            `paired with peer ${counterpart.rank}'s ${counterpart.transfer.tag}. ` +
            'The schedule generator posted transfers in an inconsistent order.',
        );
      }
      const from = p.isSend ? rank : p.transfer.peer;
      const to = p.isSend ? p.transfer.peer : rank;
      const wire = cost.transfer(p.transfer.kind, p.transfer.mb, from, to);
      // sync: data moves only after both peers posted. async: data was already
      // in flight since the sender posted, the receiver just waits for it to land.
      const done = commModel === 'sync' ? Math.max(st.clock, counterpart.time) + wire : Math.max(st.clock, counterpart.time + wire);
      if (!p.isSend) landed.set(p.transfer.tag, commModel === 'sync' ? done : counterpart.time + wire);
      if (done >= completion) {
        completion = done;
        peerArrival = commModel === 'sync' ? Math.max(st.clock, counterpart.time) : st.clock;
        last = { transfer: p.transfer, isSend: p.isSend, peerTime: counterpart.time, wire };
      }
    }
    if (completion > st.clock && last) {
      const t = last.transfer;
      idles.push({
        rank,
        start: st.clock,
        end: completion,
        reason: last.isSend ? 'wait-send' : 'wait-recv',
        detail: `${last.isSend ? 'send' : 'recv'} ${t.kind} mb${t.mb} ${last.isSend ? 'to' : 'from'} rank ${t.peer}`,
        peerWait: peerArrival - st.clock,
        transfer: last.wire,
      });
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
        const dur = cost.compute(step.kind, step.mb, step.chunk, r);
        ops.push({
          id: nextOpId++,
          rank: r,
          chunk: step.chunk,
          stage: step.chunk * pp + r,
          mb: step.mb,
          kind: step.kind,
          start: st.clock,
          end: st.clock + dur,
        });
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
    if (!unfinished) break;
    if (!progress) {
      const blocked = ranks
        .map((st, r) =>
          st.pending
            ? `rank ${r} @t=${st.clock}: ` +
              st.pending.map((p) => `${p.isSend ? 'send' : 'recv'} ${p.transfer.tag} ${p.isSend ? 'to' : 'from'} ${p.transfer.peer}`).join(', ')
            : null,
        )
        .filter((s): s is string => s !== null);
      throw new Error(`Deadlock: no rank can make progress.\n${blocked.join('\n')}`);
    }
  }

  return { ops, idles, rankFinish: ranks.map((st) => st.clock), landed };
}

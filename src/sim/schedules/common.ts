import type { CommStep, ComputeStep, OpKind, Step, Transfer } from '../types.ts';

/**
 * Helpers shared by schedule generators. All generators use Megatron's
 * interleaved placement: global stage of (chunk, rank) is `chunk * pp + rank`,
 * so a chunk's forward output flows to rank+1 (wrapping to rank 0 for the next
 * chunk) and its backward gradient flows to rank-1 (wrapping to rank pp-1).
 */
export class Topology {
  readonly pp: number;
  readonly vpp: number;
  readonly numStages: number;

  constructor(pp: number, vpp: number) {
    this.pp = pp;
    this.vpp = vpp;
    this.numStages = pp * vpp;
  }

  stage(chunk: number, rank: number): number {
    return chunk * this.pp + rank;
  }

  /** Transfer describing this rank sending its F output for (mb, chunk). Null on last stage. */
  sendF(rank: number, mb: number, chunk: number): Transfer | null {
    const s = this.stage(chunk, rank);
    if (s >= this.numStages - 1) return null;
    return { kind: 'F', peer: (rank + 1) % this.pp, tag: fTag(mb, s), mb };
  }

  /** Transfer describing this rank receiving the F input for (mb, chunk). Null on first stage. */
  recvF(rank: number, mb: number, chunk: number): Transfer | null {
    const s = this.stage(chunk, rank);
    if (s === 0) return null;
    return { kind: 'F', peer: (rank - 1 + this.pp) % this.pp, tag: fTag(mb, s - 1), mb };
  }

  /** Transfer describing this rank sending its B output (input grad) for (mb, chunk). Null on first stage. */
  sendB(rank: number, mb: number, chunk: number): Transfer | null {
    const s = this.stage(chunk, rank);
    if (s === 0) return null;
    return { kind: 'B', peer: (rank - 1 + this.pp) % this.pp, tag: bTag(mb, s), mb };
  }

  /** Transfer describing this rank receiving the B input (output grad) for (mb, chunk). Null on last stage. */
  recvB(rank: number, mb: number, chunk: number): Transfer | null {
    const s = this.stage(chunk, rank);
    if (s >= this.numStages - 1) return null;
    return { kind: 'B', peer: (rank + 1) % this.pp, tag: bTag(mb, s + 1), mb };
  }
}

/** Tag of the forward activation produced by `stage` for `mb`. */
export function fTag(mb: number, stage: number): string {
  return `F:${mb}:${stage}`;
}

/** Tag of the gradient w.r.t. the input of `stage` for `mb`. */
export function bTag(mb: number, stage: number): string {
  return `B:${mb}:${stage}`;
}

export function compute(kind: OpKind, mb: number, chunk: number): ComputeStep {
  return { type: 'compute', kind, mb, chunk };
}

/** Build a comm step from optional transfers; returns null when empty. */
export function comm(sends: (Transfer | null)[], recvs: (Transfer | null)[]): CommStep | null {
  const s = sends.filter((t): t is Transfer => t !== null);
  const r = recvs.filter((t): t is Transfer => t !== null);
  if (s.length === 0 && r.length === 0) return null;
  return { type: 'comm', sends: s, recvs: r };
}

/** Push a step if it is not null. */
export function push(steps: Step[], step: Step | null): void {
  if (step) steps.push(step);
}
